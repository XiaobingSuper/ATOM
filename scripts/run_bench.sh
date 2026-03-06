#!bin/bash

MODEL=/it-share/models/Kimi-K2-Thinking-MXFP4

RANGE_RATIO=0.8

# 1K/1K
# ISL=1000
# OSL=1000
# CON=128
# #CON=128
# NUM=$(( CON * 4 ))

# 4K/1K
#ISL=4000
#OSL=1000
#CON=128
#CON=64
#NUM=$(( CON * 4 ))

# 10K/1K
ISL=10000
OSL=1000
CON=64
# CON=32
NUM=$(( CON * 4 ))

echo "ATOM Model=${MODEL}"
echo "ATOM ISL=${ISL}, OSL=${OSL}, NUM=${NUM}, CON=${CON} RANGE_RATIO=${RANGE_RATIO}"

sleep 2

# git clone https://github.com/kimbochen/bench_serving.git
python bench_serving/benchmark_serving.py \
    --model=$MODEL \
    --backend=vllm \
    --base-url=http://localhost:8001 \
    --dataset-name=random \
    --random-input-len=$ISL \
    --random-output-len=$OSL \
    --random-range-ratio ${RANGE_RATIO} \
    --num-prompts=${NUM} \
    --max-concurrency=${CON} \
    --request-rate=inf \
    --ignore-eos \
    --percentile-metrics="ttft,tpot,itl,e2el" \
    --result-dir=./ \
    --trust-remote-code \
    --num-warmups ${NUM}
    # --profile
    # 2>&1 | tee log.bench.log

echo "ATOM Model=${MODEL}"
echo "ATOM ISL=${ISL}, OSL=${OSL}, NUM=${NUM}, CON=${CON}"
rm -rf ./*.json