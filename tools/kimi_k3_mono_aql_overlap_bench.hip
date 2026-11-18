// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Kernel-boundary overlap on one GPU: a chain of LAYERS x NODES dependent
// "stage" kernels captured in one HIP graph, each launched with
// hipExtLaunchKernelGGL, either in order (AQL barrier bit set, HIP's default)
// or with hipExtAnyOrderLaunch (barrier bit clear: the CP may start the
// packet before the previous one completes). A stage, per CTA:
//   1. streams PRO_KB of its own weights (loads that do not depend on the
//      previous stage: what an early-started kernel can do before its input),
//   2. spins until the previous stage's epoch-tagged device-scope flags carry
//      its epoch (DEP=all: every producer CTA's; DEP=one: CTA bid's; none),
//   3. reads and checks the previous stage's output at device scope,
//   4. streams BODY_KB more weights (+TAIL_KB on the last TAIL_CTAS CTAs, the
//      stragglers) and runs FMA_ITERS (+TAIL_ITERS) dependent FMAs,
//   5. stores its output at device scope, waits for the stores, then its flag.
// No acquire or release fences anywhere: the flags and the data they guard
// go past the per-XCD L2 (scope:SCOPE_DEV), as the mono kernels' st_dev/ld_dev.
//
// Each config (in-order, any-order) is its own graph over the same buffers;
// replays alternate. Per-CTA stamps (the 100 MHz realtime counter, whose
// offset differs between XCDs, so only same-XCD stamps are compared; gfx1250
// has no HW_REG_XCC_ID, a CTA's XCD is taken as bid % NXCD) give
// per boundary and XCD: the gap from the producer's last CTA end to the
// consumer's first CTA entry, how much later the consumer ends than the
// producer, and the producer's straggler duration (does an early consumer
// slow the previous kernel's tail?). Output: one JSON line per config.
//
// Build (no GPU needed):
//   hipcc -O3 --offload-arch=gfx1250 -std=c++17 \
//     tools/kimi_k3_mono_aql_overlap_bench.hip -o /tmp/aql_overlap_bench
// Run (env knobs below; defaults in parentheses):
//   GRID (CU count) NODES (8) LAYERS (4) ANY (odd | all | comma list of node
//   positions in a layer) DEP (all | one | none) PRO_KB (16) BODY_KB (64)
//   TAIL_KB (0) TAIL_CTAS (0) FMA_ITERS (0) TAIL_ITERS (0) LDS_KB (0)
//   REPS (8 alternating rounds) REPLAYS (20 replays a round) STREAM (0: graph;
//   1: direct launches, no capture) NXCD (8)
//   /tmp/aql_overlap_bench
//   DEP=none PRO_KB=0 BODY_KB=16 /tmp/aql_overlap_bench   # pure launch gap
//   LDS_KB=160 /tmp/aql_overlap_bench                     # no co-residency
//   TAIL_CTAS=8 TAIL_KB=256 /tmp/aql_overlap_bench        # straggler tail
// Check the replayed packet headers: barrier=0 on the any-order nodes (0xa02,
// or 0xa00 as a gfx1250 ext packet; 0xb02 / 0xb00 with the bit set). A ROCm
// build before 2026-07-29 (CLR #9421) drops the flag at capture:
//   AMD_LOG_LEVEL=4 AMD_LOG_MASK=0x8 REPS=1 REPLAYS=1 /tmp/aql_overlap_bench \
//     2>&1 | grep -m 40 "Dispatch Header"

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#define CHECK(x)                                                                     \
  do {                                                                               \
    hipError_t e_ = (x);                                                             \
    if (e_ != hipSuccess) {                                                          \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
      exit(1);                                                                       \
    }                                                                                \
  } while (0)

constexpr int THREADS = 256;
constexpr int VEC_BYTES = 16 * THREADS;  // one uint4 a thread
enum { S_ENTRY, S_PRO, S_SEEN, S_BODY, S_END, S_HWID, S_SPINS, STAMPS };
enum { DEP_NONE, DEP_ONE, DEP_ALL };

struct Stage {
  const int* flags_in;  // previous stage's flags, one a CTA (nullptr: no input)
  const int* data_in;   // previous stage's output, [grid_in][THREADS] words
  int* flags_out;
  int* data_out;
  int* counter;  // this stage's per-CTA launch count: its epoch
  const uint4* weights;
  unsigned long long* stamps;  // [grid][STAMPS]
  int* errors;
  unsigned* sink;
  int grid_in;
  int dep;
  int pro_vec, body_vec, tail_vec;
  int fma_iters, tail_iters, tail_ctas;
};

__device__ inline unsigned long long realtime() {
  unsigned long long t;
  asm volatile("s_sendmsg_rtn_b64 %0, sendmsg(MSG_RTN_GET_REALTIME)\n\ts_wait_kmcnt 0"
               : "=s"(t)::"memory");
  return t;
}

__device__ inline unsigned hw_id1() {
  unsigned x;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(x));
  return x;
}

__device__ inline int ld_dev(const int* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline void st_dev(int* p, int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline void wait_stores() { asm volatile("s_wait_storecnt 0x0" ::: "memory"); }

__host__ __device__ inline int tag(int epoch, int cta, int word) {
  return static_cast<int>(static_cast<unsigned>(epoch) * 1000003u + cta * THREADS + word);
}

__device__ inline void stream(const uint4* w, int first, int count, int tid, uint4& acc) {
#pragma unroll 4
  for (int i = first; i < first + count; ++i) {
    uint4 v = w[static_cast<size_t>(i) * THREADS + tid];
    acc.x ^= v.x;
    acc.y ^= v.y;
    acc.z ^= v.z;
    acc.w ^= v.w;
  }
}

__global__ __launch_bounds__(THREADS) void stage_kernel(Stage s) {
  extern __shared__ int lds[];  // sized by the launch only, to set occupancy
  __shared__ int sh_epoch;
  const int tid = threadIdx.x, bid = blockIdx.x, grid = gridDim.x;
  const unsigned long long t_entry = realtime();
  if (tid == 0) {
    const int e = ld_dev(s.counter + bid) + 1;
    st_dev(s.counter + bid, e);
    sh_epoch = e;
  }
  const bool straggler = bid >= grid - s.tail_ctas;
  const int tail_vec = straggler ? s.tail_vec : 0;
  const uint4* w =
      s.weights + static_cast<size_t>(bid) * (s.pro_vec + s.body_vec + s.tail_vec) * THREADS;
  uint4 acc = {0u, 0u, 0u, 0u};
  stream(w, 0, s.pro_vec, tid, acc);
  __syncthreads();
  const int epoch = sh_epoch;
  const unsigned long long t_pro = realtime();

  int spins = 0;
  if (s.dep == DEP_ALL) {
    for (int p = tid; p < s.grid_in; p += THREADS) {
      while (ld_dev(s.flags_in + p) < epoch) {
        __builtin_amdgcn_s_sleep(1);
        ++spins;
      }
    }
  } else if (s.dep == DEP_ONE && tid == 0) {
    while (ld_dev(s.flags_in + bid % s.grid_in) < epoch) {
      __builtin_amdgcn_s_sleep(1);
      ++spins;
    }
  }
  __syncthreads();
  const unsigned long long t_seen = realtime();

  int bad = 0;
  if (s.dep == DEP_ALL) {
    const int word = bid % THREADS;
    for (int p = tid; p < s.grid_in; p += THREADS)
      bad += ld_dev(s.data_in + static_cast<size_t>(p) * THREADS + word) != tag(epoch, p, word);
  } else if (s.dep == DEP_ONE) {
    const int p = bid % s.grid_in;
    bad += ld_dev(s.data_in + static_cast<size_t>(p) * THREADS + tid) != tag(epoch, p, tid);
  }
  if (bad) atomicAdd(s.errors, bad);

  stream(w, s.pro_vec, s.body_vec + tail_vec, tid, acc);
  float f0 = __uint_as_float(acc.x & 0x3fffffffu), f1 = f0 + 1.f, f2 = f0 + 2.f, f3 = f0 + 3.f;
  const int iters = s.fma_iters + (straggler ? s.tail_iters : 0);
  for (int i = 0; i < iters; ++i) {
    f0 = __builtin_fmaf(f0, 0.999f, 1e-3f);
    f1 = __builtin_fmaf(f1, 0.999f, 1e-3f);
    f2 = __builtin_fmaf(f2, 0.999f, 1e-3f);
    f3 = __builtin_fmaf(f3, 0.999f, 1e-3f);
  }
  const unsigned long long t_body = realtime();

  st_dev(s.data_out + static_cast<size_t>(bid) * THREADS + tid, tag(epoch, bid, tid));
  wait_stores();
  __syncthreads();
  if (tid == 0) st_dev(s.flags_out + bid, epoch);

  if ((acc.x ^ acc.y ^ acc.z ^ acc.w ^ __float_as_uint(f0 + f1 + f2 + f3)) == 0x9e3779b9u)
    s.sink[bid] = 1;
  if (tid == 0) {
    unsigned long long* st = s.stamps + static_cast<size_t>(bid) * STAMPS;
    st[S_ENTRY] = t_entry;
    st[S_PRO] = t_pro;
    st[S_SEEN] = t_seen;
    st[S_BODY] = t_body;
    st[S_HWID] = hw_id1();
    st[S_SPINS] = spins;
    st[S_END] = realtime();
  }
}

static int env_int(const char* name, int fallback) {
  const char* v = getenv(name);
  return v && *v ? atoi(v) : fallback;
}

static std::string env_str(const char* name, const char* fallback) {
  const char* v = getenv(name);
  return v && *v ? v : fallback;
}

static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1) + 0.5))];
}

struct Config {
  const char* name = "";
  std::vector<int> any;  // per chain node: 1 = launched with hipExtAnyOrderLaunch
  hipGraphExec_t exec = nullptr;
  unsigned long long* stamps = nullptr;  // [chain][grid][STAMPS]
  std::vector<double> layer_us;
};

int main() {
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  const int cus = prop.multiProcessorCount;
  const int grid = env_int("GRID", cus);
  const int nodes = env_int("NODES", 8);
  const int layers = env_int("LAYERS", 4);
  const int chain = nodes * layers;
  const std::string dep_name = env_str("DEP", "all");
  const int dep = dep_name == "none" ? DEP_NONE : dep_name == "one" ? DEP_ONE : DEP_ALL;
  const int pro_vec = env_int("PRO_KB", 16) * 1024 / VEC_BYTES;
  const int body_vec = env_int("BODY_KB", 64) * 1024 / VEC_BYTES;
  const int tail_vec = env_int("TAIL_KB", 0) * 1024 / VEC_BYTES;
  const int tail_ctas = env_int("TAIL_CTAS", 0);
  const int fma_iters = env_int("FMA_ITERS", 0);
  const int tail_iters = env_int("TAIL_ITERS", 0);
  // the kernel's static LDS (4 B) and the launch's must fit one workgroup
  const int lds_bytes = std::min<int>(env_int("LDS_KB", 0) * 1024,
                                      static_cast<int>(prop.maxSharedMemoryPerMultiProcessor) - 256);
  const int reps = env_int("REPS", 8);
  const int replays = env_int("REPLAYS", 20);
  const bool direct = env_int("STREAM", 0) != 0;
  const std::string any_spec = env_str("ANY", "odd");
  const int nxcd = env_int("NXCD", 8);  // CTA bid's XCD is bid % NXCD (round-robin placement)

  std::vector<int> any_in_layer(nodes, 0);
  if (any_spec == "all") {
    std::fill(any_in_layer.begin(), any_in_layer.end(), 1);
  } else if (any_spec == "odd") {
    for (int k = 1; k < nodes; k += 2) any_in_layer[k] = 1;
  } else {
    for (const char* p = any_spec.c_str(); *p;) {
      const int k = atoi(p);
      if (k >= 0 && k < nodes) any_in_layer[k] = 1;
      while (*p && *p != ',') ++p;
      if (*p) ++p;
    }
  }

  if (lds_bytes > 64 * 1024)
    CHECK(hipFuncSetAttribute(reinterpret_cast<const void*>(stage_kernel),
                              hipFuncAttributeMaxDynamicSharedMemorySize, lds_bytes));

  const size_t node_vec = static_cast<size_t>(grid) * (pro_vec + body_vec + tail_vec) * THREADS;
  int *flags, *data, *counters, *errors;
  unsigned* sink;
  uint4* weights;
  CHECK(hipMalloc(&flags, sizeof(int) * chain * grid));
  CHECK(hipMalloc(&data, sizeof(int) * chain * grid * THREADS));
  CHECK(hipMalloc(&counters, sizeof(int) * chain * grid));
  CHECK(hipMalloc(&errors, sizeof(int)));
  CHECK(hipMalloc(&sink, sizeof(unsigned) * grid));
  CHECK(hipMalloc(&weights, sizeof(uint4) * std::max<size_t>(node_vec * chain, 1)));
  CHECK(hipMemset(flags, 0, sizeof(int) * chain * grid));
  CHECK(hipMemset(counters, 0, sizeof(int) * chain * grid));
  CHECK(hipMemset(errors, 0, sizeof(int)));
  CHECK(hipMemset(weights, 0x5a, sizeof(uint4) * std::max<size_t>(node_vec * chain, 1)));

  Config configs[2];
  configs[0].name = "in_order";
  configs[1].name = "any_order";
  for (auto& c : configs) c.any.assign(chain, 0);
  for (int n = 1; n < chain; ++n) configs[1].any[n] = any_in_layer[n % nodes];
  for (auto& c : configs)
    CHECK(hipMalloc(&c.stamps, sizeof(unsigned long long) * chain * grid * STAMPS));

  hipStream_t stream;
  CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  auto launch_chain = [&](Config& c) {
    for (int n = 0; n < chain; ++n) {
      Stage s{};
      s.flags_in = n ? flags + (n - 1) * grid : nullptr;
      s.data_in = n ? data + static_cast<size_t>(n - 1) * grid * THREADS : nullptr;
      s.flags_out = flags + n * grid;
      s.data_out = data + static_cast<size_t>(n) * grid * THREADS;
      s.counter = counters + n * grid;
      s.weights = weights + node_vec * n;
      s.stamps = c.stamps + static_cast<size_t>(n) * grid * STAMPS;
      s.errors = errors;
      s.sink = sink;
      s.grid_in = grid;
      s.dep = n ? dep : DEP_NONE;
      s.pro_vec = pro_vec;
      s.body_vec = body_vec;
      s.tail_vec = tail_vec;
      s.fma_iters = fma_iters;
      s.tail_iters = tail_iters;
      s.tail_ctas = tail_ctas;
      hipExtLaunchKernelGGL(stage_kernel, dim3(grid), dim3(THREADS), lds_bytes, stream, nullptr,
                            nullptr, c.any[n] ? hipExtAnyOrderLaunch : 0u, s);
      CHECK(hipGetLastError());
    }
  };

  for (auto& c : configs) {
    if (direct) continue;
    hipGraph_t g;
    CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    launch_chain(c);
    CHECK(hipStreamEndCapture(stream, &g));
    CHECK(hipGraphInstantiate(&c.exec, g, nullptr, nullptr, 0));
    CHECK(hipGraphDestroy(g));
  }
  auto run = [&](Config& c) {
    if (direct)
      launch_chain(c);
    else
      CHECK(hipGraphLaunch(c.exec, stream));
  };

  for (int i = 0; i < 3; ++i)
    for (auto& c : configs) run(c);
  CHECK(hipStreamSynchronize(stream));

  hipEvent_t a, b;
  CHECK(hipEventCreate(&a));
  CHECK(hipEventCreate(&b));
  for (int r = 0; r < reps; ++r) {
    for (auto& c : configs) {
      CHECK(hipEventRecord(a, stream));
      for (int i = 0; i < replays; ++i) run(c);
      CHECK(hipEventRecord(b, stream));
      CHECK(hipEventSynchronize(b));
      float ms = 0.f;
      CHECK(hipEventElapsedTime(&ms, a, b));
      c.layer_us.push_back(1000.0 * ms / (replays * layers));
    }
  }
  // one clean replay each for the stamps (the timing loop's last one also
  // overwrote them, but stamping its own replay keeps the two independent)
  for (auto& c : configs) {
    run(c);
    CHECK(hipStreamSynchronize(stream));
  }
  int errs = 0;
  CHECK(hipMemcpy(&errs, errors, sizeof(int), hipMemcpyDeviceToHost));

  const double tick_us = 0.01;  // 100 MHz
  for (auto& c : configs) {
    std::vector<unsigned long long> st(static_cast<size_t>(chain) * grid * STAMPS);
    CHECK(hipMemcpy(st.data(), c.stamps, sizeof(unsigned long long) * st.size(),
                    hipMemcpyDeviceToHost));
    auto at = [&](int n, int cta, int k) { return st[(static_cast<size_t>(n) * grid + cta) * STAMPS + k]; };
    std::map<int, std::vector<int>> by_xcd;
    for (int cta = 0; cta < grid; ++cta) by_xcd[cta % nxcd].push_back(cta);
    // boundary n-1 -> n for every node past the first layer, split by whether
    // node n was launched any-order
    std::vector<double> gap[2], lag[2], straggler[2], span[2], wait[2];
    for (int n = nodes; n < chain; ++n) {
      const int kind = c.any[n];
      for (auto& [xcd, ctas] : by_xcd) {
        unsigned long long prev_end = 0, prev_entry = ~0ull, entry = ~0ull, end = 0;
        for (int cta : ctas) {
          prev_end = std::max(prev_end, at(n - 1, cta, S_END));
          prev_entry = std::min(prev_entry, at(n - 1, cta, S_ENTRY));
          entry = std::min(entry, at(n, cta, S_ENTRY));
          end = std::max(end, at(n, cta, S_END));
          wait[kind].push_back(tick_us * (at(n, cta, S_SEEN) - at(n, cta, S_PRO)));
        }
        gap[kind].push_back(tick_us * (static_cast<double>(entry) - static_cast<double>(prev_end)));
        lag[kind].push_back(tick_us * (static_cast<double>(end) - static_cast<double>(prev_end)));
        span[kind].push_back(tick_us * (static_cast<double>(prev_end) - static_cast<double>(prev_entry)));
      }
      for (int cta = grid - tail_ctas; cta < grid; ++cta)
        straggler[kind].push_back(tick_us * (at(n - 1, cta, S_END) - at(n - 1, cta, S_ENTRY)));
    }
    const int xcds = static_cast<int>(by_xcd.size());
    // a check of the bid % NXCD grouping: one group's entries of a node sit
    // within a few us on one clock; a group mixing XCDs shows their offsets
    double spread = 0.0;
    for (auto& [xcd, ctas] : by_xcd) {
      unsigned long long lo = ~0ull, hi = 0;
      for (int cta : ctas) {
        lo = std::min(lo, at(std::min(nodes, chain - 1), cta, S_ENTRY));
        hi = std::max(hi, at(std::min(nodes, chain - 1), cta, S_ENTRY));
      }
      spread = std::max(spread, tick_us * (hi - lo));
    }
    printf("{\"config\": \"%s\", \"cus\": %d, \"grid\": %d, \"xcds\": %d, "
           "\"xcd_entry_spread_us\": %.2f, \"nodes\": %d, "
           "\"layers\": %d, \"dep\": \"%s\", \"any\": \"%s\", \"pro_kb\": %d, \"body_kb\": %d, "
           "\"tail_kb\": %d, \"tail_ctas\": %d, \"fma_iters\": %d, \"tail_iters\": %d, "
           "\"lds_kb\": %d, \"stream\": %d, \"errors\": %d, \"layer_us_median\": %.2f, "
           "\"layer_us_min\": %.2f",
           c.name, cus, grid, xcds, spread, nodes, layers, dep_name.c_str(), any_spec.c_str(),
           pro_vec * VEC_BYTES / 1024, body_vec * VEC_BYTES / 1024, tail_vec * VEC_BYTES / 1024,
           tail_ctas, fma_iters, tail_iters, lds_bytes / 1024, direct ? 1 : 0, errs,
           median(c.layer_us), pct(c.layer_us, 0.0));
    const char* kinds[2] = {"in_order_boundaries", "any_order_boundaries"};
    for (int kind = 0; kind < 2; ++kind) {
      if (gap[kind].empty()) continue;
      printf(", \"%s\": {\"count\": %zu, \"gap_us_p50\": %.3f, \"gap_us_p10\": %.3f, "
             "\"gap_us_p90\": %.3f, \"lag_us_p50\": %.3f, \"producer_span_us_p50\": %.3f, "
             "\"consumer_wait_us_p50\": %.3f, \"straggler_us_p50\": %.3f}",
             kinds[kind], gap[kind].size() / std::max(xcds, 1), median(gap[kind]),
             pct(gap[kind], 0.1), pct(gap[kind], 0.9), median(lag[kind]), median(span[kind]),
             median(wait[kind]), median(straggler[kind]));
    }
    printf("}\n");
  }
  return errs != 0;
}
