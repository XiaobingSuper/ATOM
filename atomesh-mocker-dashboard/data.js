window.BENCHMARK_DATA = {
  "lastUpdate": 1795497343972,
  "repoUrl": "https://github.com/XiaobingSuper/ATOM",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "zufa.yu@amd.com",
            "name": "zufayu",
            "username": "zufayu"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ec17e6f8df7b54d47e9e951651265ab049af39ae",
          "message": "review-pr: publish the coverage verdict, and fail the gate when it cannot be checked (#3456)\n\n* review-pr: publish the coverage verdict, not just compute it\n\ntriage.py measures whether a diff fits one review pass, SKILL.md requires the\ncard to carry it, and _check_coverage_line gates that the card does. All three\nhave been in place since the port. No published review has ever carried the\nline.\n\nThe gate reads card.md. build_review assembles the body that is posted, and\nits prefix list named only \"Review:\" and \"Validation\". So the measurement was\ntaken, required, checked, and then dropped on the way out -- with every gate\ngreen, because they were watching the other copy.\n\nTwenty-one reviews went out this way. The one that mattered most was a\n10179-line, 54-file diff: it got two findings and no statement that two\nfindings on that much code are a sample rather than a sweep. _check_coverage_line's\nown docstring names this reader -- \"a reader cannot tell a card that omitted the\nline from one whose diff was small enough not to need it\" -- and that is exactly\nwho has been reading these.\n\nThe test fixture gains a Coverage line (asserting against an input that lacks\none proves nothing) and the new test pins both that it is published and that it\nsits above the findings index. Dropping \"Coverage:\" from the prefix list again\nturns it red.\n\ncheck_prompts.py 0 drift; selftest.sh 115 green; pytest 45 passed; black and\nruff clean.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): the coverage gate returned green when it read nothing\n\n`_check_coverage_line` compared the card's `Coverage:` line against the one\nStep 1 wrote into rules.txt. Three ways of reading nothing all returned `[]`,\nwhich the card gate cannot tell apart from a clean comparison:\n\n  - rules.txt missing or unreadable (`except OSError: return []`)\n  - rules.txt present but not Step 1's output (no `coverage:` line)\n  - `[diff-too-large]` -- GitHub refusing to serve a diff over 20000 lines\n\nThe third is the expensive one. That marker means the largest PR this bot can\never see, the exact case where \"the findings below are a sample, not a sweep\"\nhas to reach the reader -- and the card could omit coverage entirely and pass.\nA check that does not run looked exactly like a check that passed.\n\nNow: an unreadable or non-Step-1 rules.txt raises UNCHECKED-COVERAGE, and\n`[diff-too-large]` requires a coverage line on the card. `[underivable]` and\n`[derivation-failed]` keep returning `[]` and are named as the only entitled\nsilence: there Step 1 measured nothing, so no verdict exists for the card to\ndrop.\n\nSix selftest cases, four of them mutation-verified -- silencing any of the\nthree new paths, or widening the entitled-silence check to swallow\n`[diff-too-large]`, turns tests red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): say when a review could anchor nothing\n\nThe summary names the inline count only when there is one. With no anchors at\nall the block vanishes, so the review goes out with no inline comments and\nnothing saying any were expected.\n\nATOM#2152 -- 4546 lines over 8 files, the largest diff this bot has reviewed --\npublished six findings that way. Five cite no `path:line`; the sixth cites\nmerge-target files (`moe.py`, `loader.py`) the diff never touches, so\n`resolve_path` finds nothing to attach to. Six real findings, zero anchors,\nzero signal that the inline half of the review did not happen.\n\nThe surviving branch already argues this case against itself: \"an invisible\nfinding reads exactly like a finding that was never made.\" It stops one\ncondition short of its own reasoning -- partial invisibility is called out,\ntotal invisibility is not.\n\nNow a review with findings and no anchors says `0 of N finding(s) could be\nanchored to a line in this diff`, and says what anchoring needs. Mutation-\nverified: silencing the branch turns the test red. Dry-run against #2152's\nreal card emits exactly that line. 42 passed, 121 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* style: black the new anchor test\n\npsf/black@stable collapses the two-line .replace() call added with the\n0-anchor test. Formatting only.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): a diff GitHub refused to serve may not be called ordinary\n\nReview feedback on the previous commit: the `[diff-too-large]` branch checked\nonly that SOME coverage line was present, never what it said. So the softening\nthe measured path exists to refuse -- reporting a BEYOND ONE PASS diff as an\nordinary one -- walked straight through on the one diff large enough that\nGitHub would not serve it at all.\n\nOver 20000 lines is an order of magnitude past BIG_ADDED, so such a diff is\nBEYOND ONE PASS by definition and the card has to say the words.\n\nThe same review asked what the card should say when rules.txt has no line to\ncopy. Both messages now spell the line out verbatim instead of describing it.\n\nSeventh selftest case; mutation-verified (restoring the presence-only check\nturns it red). 122 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): match the coverage line the way the gate does, and drop two redundant reds\n\nReview feedback on 9113fd2 (all three verified before acting).\n\n1. The publisher reintroduced this PR's own bug one case over. The gate accepts\n   any line whose `strip().lower()` starts `coverage:`; `_line_starting` required\n   the exact prefix `Coverage:` at column 0. A card copying rules.txt's\n   `  coverage: ...` verbatim -- which SKILL.md's template asks for -- passed the\n   gate and lost the line on the way out. The new `[diff-too-large]` message made\n   it certain by spelling the hint in lower case. `_line_starting` now strips and\n   folds case, so gate and publisher accept the same line; the hint is written in\n   the card's own `Coverage:` form. Three new publisher cases (indented, lower\n   case, both).\n\n2. The two UNCHECKED-COVERAGE branches were a second red for a state already\n   blocked: the ledger gate (SKILL.md:380) opens rules.txt with no guard and exits\n   1 on LEDGER UNUSABLE, both before the card gate. Dropped, and the\n   `[diff-too-large]` case folded into the single card-line scan instead of a\n   second copy of the loop -- 72 lines to 44. Their selftest cases are replaced by\n   two that assert the ledger gate's exit code directly, because deleting a check\n   is only safe while the check replacing it is the one being measured.\n\n3. EMPTY-COVERAGE had no case at all: removing the branch left every test green.\n   Added one.\n\nComments at each site trimmed to what the code guarantees; the incident history\nlives in the commit log.\n\n124 green / 0 red, 43 publisher tests. Mutation-verified: restoring the strict\nprefix, dropping the `[diff-too-large]` branch, dropping the softening check, or\ndropping the empty check each turns tests red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: assert the anchor count, and finish the comment trim\n\nReview feedback on d177c3f (both items verified first).\n\n1. The anchor tests did not pin what they check. `test_a_review_that_anchors_nothing_says_so`\n   matched only the phrase, so dropping the count stayed green; nothing asserted the\n   line is ABSENT when some findings do anchor, so `elif loose:` degrading to\n   `if loose:` -- which emits the index block and the 0-of-N line together -- stayed\n   green too. Confirmed by running both mutations. Now the first asserts\n   `0 of 2 finding(s) could be anchored` (two of the fixture's findings fail to anchor,\n   not one) and the mixed-anchor test asserts the line is absent. Four mutations now\n   turn tests red: `elif`->`if`, dropping the count, silencing the branch, and\n   restoring `_line_starting`'s strict prefix.\n\n2. The previous commit message said the comments were trimmed; only triage.py was.\n   Trimmed the rest: _publish.py's coverage and 0-anchor blocks, selftest.sh's two\n   headers, and three test docstrings. Each now says what the code guarantees.\n\nAlso noted in review and confirmed here: fetch.sh:53-64 exits 1 when GitHub refuses a\ndiff over 20000 lines, so `[diff-too-large]` never reaches rules.txt in the run_one\npipeline. The branch is a backstop for a hand-run triage, not a live path; the PR\ndescription no longer calls it \"the surviving case\".\n\n124 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: call the diff-too-large rule a policy, and finish the trim\n\nReview feedback on f984656 (verified before acting).\n\nThe `[diff-too-large]` comment claimed a 20000-line refusal is \"an order of\nmagnitude past BIG_ADDED, so BEYOND ONE PASS by definition\". That does not\nfollow, and the review is right: GitHub's cap counts every line of the diff\n(fetch.sh), while BIG_ADDED counts only non-test ADDED lines (triage.py:1236-38,\nBIG_ADDED = 500). A deletion-heavy or test-heavy refusal could measure as one\npass. The comment now says BEYOND is required by policy, and gives the reason\nthat does hold: the diff was never served, so a review of what could not be read\nis a sample whatever the counts would have said. Behaviour is unchanged.\n\nLast two history sites trimmed: `_line_starting`'s docstring and the\n`[diff-too-large]` selftest header.\n\n124 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: say policy in the message too, and describe anchoring exactly\n\nReview feedback on c119eb7 (both verified).\n\n1. The previous commit made the `[diff-too-large]` COMMENT say \"policy, not\n   measurement\" and left the SOFTENED-COVERAGE MESSAGE opening \"Step 1 measured\n   this diff as BEYOND ONE PASS\" -- in the one case where no measurement exists.\n   The opening clause now branches on `too_large`: a refused diff gets \"GitHub\n   would not serve this diff, so it counts as BEYOND ONE PASS\", a measured one\n   keeps its wording. Two selftest cases pin both halves, mutation-verified in\n   both directions.\n\n2. The 0-of-N notice said a finding anchors when it cites a line \"the PR\n   actually changed\". `commentable()` records added AND context lines inside a\n   hunk, so \"inside this diff's hunks\" is what is true.\n\n126 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: drop a stray f-prefix and put the spelled line last\n\nCosmetic, from review. The first literal of the `why` expression carried an\n`f` with no placeholder, and the refused-diff message ended with the spelled\ncoverage line and then repeated \"sample\" after it. The suffix now comes last,\nso the message reads reason, consequence, then what to write.\n\nNo behaviour change. 126 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: cover three branches a mutation sweep found untested\n\nNot from review -- from mutating every logical line this PR adds and checking\nwhich mutations stayed green. Four did. Three were real gaps:\n\n1. The MISSING-COVERAGE reason could claim a measurement that does not exist.\n   This is the same defect review caught in SOFTENED-COVERAGE one commit ago,\n   and the fix there covered only that branch: swapping MISSING's `too_large`\n   reason for \"Step 1 measured the diff\" kept every test green.\n\n2. MISSING-COVERAGE's text field falls back to `[diff-too-large]` when there is\n   no `coverage:` line to name. Dropping the fallback printed an empty field and\n   nothing noticed.\n\n3. `too_large` is guarded by `not want`, so a rules.txt carrying both a real\n   coverage line and the marker compares against the line rather than falling\n   back to the refusal policy. A measurement outranks a policy; nothing pinned\n   that.\n\nThree selftest cases, each mutation-verified.\n\nThe fourth surviving mutation is not a defect: the rules.txt scan is\ncase-sensitive while the card scan is not, which is the right asymmetry --\nstrict about this module's own output, liberal about what the model writes.\nCommented rather than tested, so it does not read as an oversight.\n\n129 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: zufayu <8138883+zufayu@users.noreply.github.com>\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-11-18T12:41:15+08:00",
          "tree_id": "9c864ded1dfffce9b50c542b21a95aa221085a73",
          "url": "https://github.com/XiaobingSuper/ATOM/commit/ec17e6f8df7b54d47e9e951651265ab049af39ae"
        },
        "date": 1794983572595,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 3047.68,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=548582 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.31,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=548582 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.35,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=548582 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.66,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=548582 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=548582 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 9958.82,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1792588 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.57,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1792588 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 2.65,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1792588 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 3.51,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1792588 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1792588 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 5082.94,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=914930 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.37,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=914930 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=914930 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.59,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=914930 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=914930 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 7997.91,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1439623 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1439623 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1439623 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.07,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1439623 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1439623 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 9636.87,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1734636 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.8,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1734636 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.39,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1734636 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 1.89,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1734636 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1734636 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 3007.32,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=541317 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.32,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=541317 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=541317 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.83,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=541317 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=541317 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 9899.94,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1781990 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.58,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1781990 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 2.75,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1781990 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.18,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1781990 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1781990 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 5062.96,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=911332 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.37,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=911332 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=911332 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.93,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=911332 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=911332 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 7564.42,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1361595 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1361595 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.94,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1361595 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 2.07,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1361595 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1361595 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 9074.05,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1633329 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.85,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1633329 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.7,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1633329 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 3.62,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1633329 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1633329 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 3029.94,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=545390 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.32,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=545390 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=545390 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.7,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=545390 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=545390 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 9987.44,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1797739 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.57,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1797739 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 2.64,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1797739 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 3.43,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1797739 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1797739 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 5091.78,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=916521 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.37,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=916521 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=916521 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.6,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=916521 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=916521 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 7936.81,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1428626 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1428626 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1428626 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1428626 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1428626 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 9678,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1742040 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.8,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1742040 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.39,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1742040 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 1.88,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1742040 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1742040 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44243299270"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zufayu",
            "username": "zufayu",
            "email": "zufa.yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ec17e6f8df7b54d47e9e951651265ab049af39ae",
          "message": "review-pr: publish the coverage verdict, and fail the gate when it cannot be checked (#3456)\n\n* review-pr: publish the coverage verdict, not just compute it\n\ntriage.py measures whether a diff fits one review pass, SKILL.md requires the\ncard to carry it, and _check_coverage_line gates that the card does. All three\nhave been in place since the port. No published review has ever carried the\nline.\n\nThe gate reads card.md. build_review assembles the body that is posted, and\nits prefix list named only \"Review:\" and \"Validation\". So the measurement was\ntaken, required, checked, and then dropped on the way out -- with every gate\ngreen, because they were watching the other copy.\n\nTwenty-one reviews went out this way. The one that mattered most was a\n10179-line, 54-file diff: it got two findings and no statement that two\nfindings on that much code are a sample rather than a sweep. _check_coverage_line's\nown docstring names this reader -- \"a reader cannot tell a card that omitted the\nline from one whose diff was small enough not to need it\" -- and that is exactly\nwho has been reading these.\n\nThe test fixture gains a Coverage line (asserting against an input that lacks\none proves nothing) and the new test pins both that it is published and that it\nsits above the findings index. Dropping \"Coverage:\" from the prefix list again\nturns it red.\n\ncheck_prompts.py 0 drift; selftest.sh 115 green; pytest 45 passed; black and\nruff clean.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): the coverage gate returned green when it read nothing\n\n`_check_coverage_line` compared the card's `Coverage:` line against the one\nStep 1 wrote into rules.txt. Three ways of reading nothing all returned `[]`,\nwhich the card gate cannot tell apart from a clean comparison:\n\n  - rules.txt missing or unreadable (`except OSError: return []`)\n  - rules.txt present but not Step 1's output (no `coverage:` line)\n  - `[diff-too-large]` -- GitHub refusing to serve a diff over 20000 lines\n\nThe third is the expensive one. That marker means the largest PR this bot can\never see, the exact case where \"the findings below are a sample, not a sweep\"\nhas to reach the reader -- and the card could omit coverage entirely and pass.\nA check that does not run looked exactly like a check that passed.\n\nNow: an unreadable or non-Step-1 rules.txt raises UNCHECKED-COVERAGE, and\n`[diff-too-large]` requires a coverage line on the card. `[underivable]` and\n`[derivation-failed]` keep returning `[]` and are named as the only entitled\nsilence: there Step 1 measured nothing, so no verdict exists for the card to\ndrop.\n\nSix selftest cases, four of them mutation-verified -- silencing any of the\nthree new paths, or widening the entitled-silence check to swallow\n`[diff-too-large]`, turns tests red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): say when a review could anchor nothing\n\nThe summary names the inline count only when there is one. With no anchors at\nall the block vanishes, so the review goes out with no inline comments and\nnothing saying any were expected.\n\nATOM#2152 -- 4546 lines over 8 files, the largest diff this bot has reviewed --\npublished six findings that way. Five cite no `path:line`; the sixth cites\nmerge-target files (`moe.py`, `loader.py`) the diff never touches, so\n`resolve_path` finds nothing to attach to. Six real findings, zero anchors,\nzero signal that the inline half of the review did not happen.\n\nThe surviving branch already argues this case against itself: \"an invisible\nfinding reads exactly like a finding that was never made.\" It stops one\ncondition short of its own reasoning -- partial invisibility is called out,\ntotal invisibility is not.\n\nNow a review with findings and no anchors says `0 of N finding(s) could be\nanchored to a line in this diff`, and says what anchoring needs. Mutation-\nverified: silencing the branch turns the test red. Dry-run against #2152's\nreal card emits exactly that line. 42 passed, 121 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* style: black the new anchor test\n\npsf/black@stable collapses the two-line .replace() call added with the\n0-anchor test. Formatting only.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): a diff GitHub refused to serve may not be called ordinary\n\nReview feedback on the previous commit: the `[diff-too-large]` branch checked\nonly that SOME coverage line was present, never what it said. So the softening\nthe measured path exists to refuse -- reporting a BEYOND ONE PASS diff as an\nordinary one -- walked straight through on the one diff large enough that\nGitHub would not serve it at all.\n\nOver 20000 lines is an order of magnitude past BIG_ADDED, so such a diff is\nBEYOND ONE PASS by definition and the card has to say the words.\n\nThe same review asked what the card should say when rules.txt has no line to\ncopy. Both messages now spell the line out verbatim instead of describing it.\n\nSeventh selftest case; mutation-verified (restoring the presence-only check\nturns it red). 122 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): match the coverage line the way the gate does, and drop two redundant reds\n\nReview feedback on 9113fd2 (all three verified before acting).\n\n1. The publisher reintroduced this PR's own bug one case over. The gate accepts\n   any line whose `strip().lower()` starts `coverage:`; `_line_starting` required\n   the exact prefix `Coverage:` at column 0. A card copying rules.txt's\n   `  coverage: ...` verbatim -- which SKILL.md's template asks for -- passed the\n   gate and lost the line on the way out. The new `[diff-too-large]` message made\n   it certain by spelling the hint in lower case. `_line_starting` now strips and\n   folds case, so gate and publisher accept the same line; the hint is written in\n   the card's own `Coverage:` form. Three new publisher cases (indented, lower\n   case, both).\n\n2. The two UNCHECKED-COVERAGE branches were a second red for a state already\n   blocked: the ledger gate (SKILL.md:380) opens rules.txt with no guard and exits\n   1 on LEDGER UNUSABLE, both before the card gate. Dropped, and the\n   `[diff-too-large]` case folded into the single card-line scan instead of a\n   second copy of the loop -- 72 lines to 44. Their selftest cases are replaced by\n   two that assert the ledger gate's exit code directly, because deleting a check\n   is only safe while the check replacing it is the one being measured.\n\n3. EMPTY-COVERAGE had no case at all: removing the branch left every test green.\n   Added one.\n\nComments at each site trimmed to what the code guarantees; the incident history\nlives in the commit log.\n\n124 green / 0 red, 43 publisher tests. Mutation-verified: restoring the strict\nprefix, dropping the `[diff-too-large]` branch, dropping the softening check, or\ndropping the empty check each turns tests red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: assert the anchor count, and finish the comment trim\n\nReview feedback on d177c3f (both items verified first).\n\n1. The anchor tests did not pin what they check. `test_a_review_that_anchors_nothing_says_so`\n   matched only the phrase, so dropping the count stayed green; nothing asserted the\n   line is ABSENT when some findings do anchor, so `elif loose:` degrading to\n   `if loose:` -- which emits the index block and the 0-of-N line together -- stayed\n   green too. Confirmed by running both mutations. Now the first asserts\n   `0 of 2 finding(s) could be anchored` (two of the fixture's findings fail to anchor,\n   not one) and the mixed-anchor test asserts the line is absent. Four mutations now\n   turn tests red: `elif`->`if`, dropping the count, silencing the branch, and\n   restoring `_line_starting`'s strict prefix.\n\n2. The previous commit message said the comments were trimmed; only triage.py was.\n   Trimmed the rest: _publish.py's coverage and 0-anchor blocks, selftest.sh's two\n   headers, and three test docstrings. Each now says what the code guarantees.\n\nAlso noted in review and confirmed here: fetch.sh:53-64 exits 1 when GitHub refuses a\ndiff over 20000 lines, so `[diff-too-large]` never reaches rules.txt in the run_one\npipeline. The branch is a backstop for a hand-run triage, not a live path; the PR\ndescription no longer calls it \"the surviving case\".\n\n124 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: call the diff-too-large rule a policy, and finish the trim\n\nReview feedback on f984656 (verified before acting).\n\nThe `[diff-too-large]` comment claimed a 20000-line refusal is \"an order of\nmagnitude past BIG_ADDED, so BEYOND ONE PASS by definition\". That does not\nfollow, and the review is right: GitHub's cap counts every line of the diff\n(fetch.sh), while BIG_ADDED counts only non-test ADDED lines (triage.py:1236-38,\nBIG_ADDED = 500). A deletion-heavy or test-heavy refusal could measure as one\npass. The comment now says BEYOND is required by policy, and gives the reason\nthat does hold: the diff was never served, so a review of what could not be read\nis a sample whatever the counts would have said. Behaviour is unchanged.\n\nLast two history sites trimmed: `_line_starting`'s docstring and the\n`[diff-too-large]` selftest header.\n\n124 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: say policy in the message too, and describe anchoring exactly\n\nReview feedback on c119eb7 (both verified).\n\n1. The previous commit made the `[diff-too-large]` COMMENT say \"policy, not\n   measurement\" and left the SOFTENED-COVERAGE MESSAGE opening \"Step 1 measured\n   this diff as BEYOND ONE PASS\" -- in the one case where no measurement exists.\n   The opening clause now branches on `too_large`: a refused diff gets \"GitHub\n   would not serve this diff, so it counts as BEYOND ONE PASS\", a measured one\n   keeps its wording. Two selftest cases pin both halves, mutation-verified in\n   both directions.\n\n2. The 0-of-N notice said a finding anchors when it cites a line \"the PR\n   actually changed\". `commentable()` records added AND context lines inside a\n   hunk, so \"inside this diff's hunks\" is what is true.\n\n126 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: drop a stray f-prefix and put the spelled line last\n\nCosmetic, from review. The first literal of the `why` expression carried an\n`f` with no placeholder, and the refused-diff message ended with the spelled\ncoverage line and then repeated \"sample\" after it. The suffix now comes last,\nso the message reads reason, consequence, then what to write.\n\nNo behaviour change. 126 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: cover three branches a mutation sweep found untested\n\nNot from review -- from mutating every logical line this PR adds and checking\nwhich mutations stayed green. Four did. Three were real gaps:\n\n1. The MISSING-COVERAGE reason could claim a measurement that does not exist.\n   This is the same defect review caught in SOFTENED-COVERAGE one commit ago,\n   and the fix there covered only that branch: swapping MISSING's `too_large`\n   reason for \"Step 1 measured the diff\" kept every test green.\n\n2. MISSING-COVERAGE's text field falls back to `[diff-too-large]` when there is\n   no `coverage:` line to name. Dropping the fallback printed an empty field and\n   nothing noticed.\n\n3. `too_large` is guarded by `not want`, so a rules.txt carrying both a real\n   coverage line and the marker compares against the line rather than falling\n   back to the refusal policy. A measurement outranks a policy; nothing pinned\n   that.\n\nThree selftest cases, each mutation-verified.\n\nThe fourth surviving mutation is not a defect: the rules.txt scan is\ncase-sensitive while the card scan is not, which is the right asymmetry --\nstrict about this module's own output, liberal about what the model writes.\nCommented rather than tested, so it does not read as an oversight.\n\n129 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: zufayu <8138883+zufayu@users.noreply.github.com>\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-11-18T04:41:15Z",
          "url": "https://github.com/XiaobingSuper/ATOM/commit/ec17e6f8df7b54d47e9e951651265ab049af39ae"
        },
        "date": 1795051388642,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2324.63,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418433 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418433 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418433 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418433 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=418433 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7231.32,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1301638 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.17,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1301638 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.66,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1301638 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.6,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1301638 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1301638 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3835.77,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690439 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690439 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690439 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690439 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=690439 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5812.14,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1046186 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1046186 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1046186 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.4,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1046186 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1046186 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6847.5,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1232550 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.12,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1232550 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.95,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1232550 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1232550 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1232550 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2338.93,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=421007 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=421007 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=421007 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=421007 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=421007 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7156.83,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1288230 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.19,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1288230 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.68,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1288230 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.59,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1288230 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1288230 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3883.42,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699015 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699015 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.62,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699015 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699015 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699015 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5918.41,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1065314 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.64,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1065314 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.03,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1065314 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.36,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1065314 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1065314 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6893.87,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1240896 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.12,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1240896 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.93,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1240896 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.52,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1240896 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1240896 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2329.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=419281 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=419281 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=419281 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=419281 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=419281 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7142.47,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1285645 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.19,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1285645 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.68,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1285645 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.61,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1285645 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1285645 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3859.41,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694693 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694693 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694693 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694693 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=694693 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5908.88,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1063598 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.64,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1063598 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.03,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1063598 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.38,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1063598 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1063598 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6949.34,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1250882 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.11,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1250882 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.93,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1250882 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.5,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1250882 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1250882 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44426277992"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zufayu",
            "username": "zufayu",
            "email": "zufa.yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ec17e6f8df7b54d47e9e951651265ab049af39ae",
          "message": "review-pr: publish the coverage verdict, and fail the gate when it cannot be checked (#3456)\n\n* review-pr: publish the coverage verdict, not just compute it\n\ntriage.py measures whether a diff fits one review pass, SKILL.md requires the\ncard to carry it, and _check_coverage_line gates that the card does. All three\nhave been in place since the port. No published review has ever carried the\nline.\n\nThe gate reads card.md. build_review assembles the body that is posted, and\nits prefix list named only \"Review:\" and \"Validation\". So the measurement was\ntaken, required, checked, and then dropped on the way out -- with every gate\ngreen, because they were watching the other copy.\n\nTwenty-one reviews went out this way. The one that mattered most was a\n10179-line, 54-file diff: it got two findings and no statement that two\nfindings on that much code are a sample rather than a sweep. _check_coverage_line's\nown docstring names this reader -- \"a reader cannot tell a card that omitted the\nline from one whose diff was small enough not to need it\" -- and that is exactly\nwho has been reading these.\n\nThe test fixture gains a Coverage line (asserting against an input that lacks\none proves nothing) and the new test pins both that it is published and that it\nsits above the findings index. Dropping \"Coverage:\" from the prefix list again\nturns it red.\n\ncheck_prompts.py 0 drift; selftest.sh 115 green; pytest 45 passed; black and\nruff clean.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): the coverage gate returned green when it read nothing\n\n`_check_coverage_line` compared the card's `Coverage:` line against the one\nStep 1 wrote into rules.txt. Three ways of reading nothing all returned `[]`,\nwhich the card gate cannot tell apart from a clean comparison:\n\n  - rules.txt missing or unreadable (`except OSError: return []`)\n  - rules.txt present but not Step 1's output (no `coverage:` line)\n  - `[diff-too-large]` -- GitHub refusing to serve a diff over 20000 lines\n\nThe third is the expensive one. That marker means the largest PR this bot can\never see, the exact case where \"the findings below are a sample, not a sweep\"\nhas to reach the reader -- and the card could omit coverage entirely and pass.\nA check that does not run looked exactly like a check that passed.\n\nNow: an unreadable or non-Step-1 rules.txt raises UNCHECKED-COVERAGE, and\n`[diff-too-large]` requires a coverage line on the card. `[underivable]` and\n`[derivation-failed]` keep returning `[]` and are named as the only entitled\nsilence: there Step 1 measured nothing, so no verdict exists for the card to\ndrop.\n\nSix selftest cases, four of them mutation-verified -- silencing any of the\nthree new paths, or widening the entitled-silence check to swallow\n`[diff-too-large]`, turns tests red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): say when a review could anchor nothing\n\nThe summary names the inline count only when there is one. With no anchors at\nall the block vanishes, so the review goes out with no inline comments and\nnothing saying any were expected.\n\nATOM#2152 -- 4546 lines over 8 files, the largest diff this bot has reviewed --\npublished six findings that way. Five cite no `path:line`; the sixth cites\nmerge-target files (`moe.py`, `loader.py`) the diff never touches, so\n`resolve_path` finds nothing to attach to. Six real findings, zero anchors,\nzero signal that the inline half of the review did not happen.\n\nThe surviving branch already argues this case against itself: \"an invisible\nfinding reads exactly like a finding that was never made.\" It stops one\ncondition short of its own reasoning -- partial invisibility is called out,\ntotal invisibility is not.\n\nNow a review with findings and no anchors says `0 of N finding(s) could be\nanchored to a line in this diff`, and says what anchoring needs. Mutation-\nverified: silencing the branch turns the test red. Dry-run against #2152's\nreal card emits exactly that line. 42 passed, 121 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* style: black the new anchor test\n\npsf/black@stable collapses the two-line .replace() call added with the\n0-anchor test. Formatting only.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): a diff GitHub refused to serve may not be called ordinary\n\nReview feedback on the previous commit: the `[diff-too-large]` branch checked\nonly that SOME coverage line was present, never what it said. So the softening\nthe measured path exists to refuse -- reporting a BEYOND ONE PASS diff as an\nordinary one -- walked straight through on the one diff large enough that\nGitHub would not serve it at all.\n\nOver 20000 lines is an order of magnitude past BIG_ADDED, so such a diff is\nBEYOND ONE PASS by definition and the card has to say the words.\n\nThe same review asked what the card should say when rules.txt has no line to\ncopy. Both messages now spell the line out verbatim instead of describing it.\n\nSeventh selftest case; mutation-verified (restoring the presence-only check\nturns it red). 122 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): match the coverage line the way the gate does, and drop two redundant reds\n\nReview feedback on 9113fd2 (all three verified before acting).\n\n1. The publisher reintroduced this PR's own bug one case over. The gate accepts\n   any line whose `strip().lower()` starts `coverage:`; `_line_starting` required\n   the exact prefix `Coverage:` at column 0. A card copying rules.txt's\n   `  coverage: ...` verbatim -- which SKILL.md's template asks for -- passed the\n   gate and lost the line on the way out. The new `[diff-too-large]` message made\n   it certain by spelling the hint in lower case. `_line_starting` now strips and\n   folds case, so gate and publisher accept the same line; the hint is written in\n   the card's own `Coverage:` form. Three new publisher cases (indented, lower\n   case, both).\n\n2. The two UNCHECKED-COVERAGE branches were a second red for a state already\n   blocked: the ledger gate (SKILL.md:380) opens rules.txt with no guard and exits\n   1 on LEDGER UNUSABLE, both before the card gate. Dropped, and the\n   `[diff-too-large]` case folded into the single card-line scan instead of a\n   second copy of the loop -- 72 lines to 44. Their selftest cases are replaced by\n   two that assert the ledger gate's exit code directly, because deleting a check\n   is only safe while the check replacing it is the one being measured.\n\n3. EMPTY-COVERAGE had no case at all: removing the branch left every test green.\n   Added one.\n\nComments at each site trimmed to what the code guarantees; the incident history\nlives in the commit log.\n\n124 green / 0 red, 43 publisher tests. Mutation-verified: restoring the strict\nprefix, dropping the `[diff-too-large]` branch, dropping the softening check, or\ndropping the empty check each turns tests red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: assert the anchor count, and finish the comment trim\n\nReview feedback on d177c3f (both items verified first).\n\n1. The anchor tests did not pin what they check. `test_a_review_that_anchors_nothing_says_so`\n   matched only the phrase, so dropping the count stayed green; nothing asserted the\n   line is ABSENT when some findings do anchor, so `elif loose:` degrading to\n   `if loose:` -- which emits the index block and the 0-of-N line together -- stayed\n   green too. Confirmed by running both mutations. Now the first asserts\n   `0 of 2 finding(s) could be anchored` (two of the fixture's findings fail to anchor,\n   not one) and the mixed-anchor test asserts the line is absent. Four mutations now\n   turn tests red: `elif`->`if`, dropping the count, silencing the branch, and\n   restoring `_line_starting`'s strict prefix.\n\n2. The previous commit message said the comments were trimmed; only triage.py was.\n   Trimmed the rest: _publish.py's coverage and 0-anchor blocks, selftest.sh's two\n   headers, and three test docstrings. Each now says what the code guarantees.\n\nAlso noted in review and confirmed here: fetch.sh:53-64 exits 1 when GitHub refuses a\ndiff over 20000 lines, so `[diff-too-large]` never reaches rules.txt in the run_one\npipeline. The branch is a backstop for a hand-run triage, not a live path; the PR\ndescription no longer calls it \"the surviving case\".\n\n124 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: call the diff-too-large rule a policy, and finish the trim\n\nReview feedback on f984656 (verified before acting).\n\nThe `[diff-too-large]` comment claimed a 20000-line refusal is \"an order of\nmagnitude past BIG_ADDED, so BEYOND ONE PASS by definition\". That does not\nfollow, and the review is right: GitHub's cap counts every line of the diff\n(fetch.sh), while BIG_ADDED counts only non-test ADDED lines (triage.py:1236-38,\nBIG_ADDED = 500). A deletion-heavy or test-heavy refusal could measure as one\npass. The comment now says BEYOND is required by policy, and gives the reason\nthat does hold: the diff was never served, so a review of what could not be read\nis a sample whatever the counts would have said. Behaviour is unchanged.\n\nLast two history sites trimmed: `_line_starting`'s docstring and the\n`[diff-too-large]` selftest header.\n\n124 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: say policy in the message too, and describe anchoring exactly\n\nReview feedback on c119eb7 (both verified).\n\n1. The previous commit made the `[diff-too-large]` COMMENT say \"policy, not\n   measurement\" and left the SOFTENED-COVERAGE MESSAGE opening \"Step 1 measured\n   this diff as BEYOND ONE PASS\" -- in the one case where no measurement exists.\n   The opening clause now branches on `too_large`: a refused diff gets \"GitHub\n   would not serve this diff, so it counts as BEYOND ONE PASS\", a measured one\n   keeps its wording. Two selftest cases pin both halves, mutation-verified in\n   both directions.\n\n2. The 0-of-N notice said a finding anchors when it cites a line \"the PR\n   actually changed\". `commentable()` records added AND context lines inside a\n   hunk, so \"inside this diff's hunks\" is what is true.\n\n126 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: drop a stray f-prefix and put the spelled line last\n\nCosmetic, from review. The first literal of the `why` expression carried an\n`f` with no placeholder, and the refused-diff message ended with the spelled\ncoverage line and then repeated \"sample\" after it. The suffix now comes last,\nso the message reads reason, consequence, then what to write.\n\nNo behaviour change. 126 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: cover three branches a mutation sweep found untested\n\nNot from review -- from mutating every logical line this PR adds and checking\nwhich mutations stayed green. Four did. Three were real gaps:\n\n1. The MISSING-COVERAGE reason could claim a measurement that does not exist.\n   This is the same defect review caught in SOFTENED-COVERAGE one commit ago,\n   and the fix there covered only that branch: swapping MISSING's `too_large`\n   reason for \"Step 1 measured the diff\" kept every test green.\n\n2. MISSING-COVERAGE's text field falls back to `[diff-too-large]` when there is\n   no `coverage:` line to name. Dropping the fallback printed an empty field and\n   nothing noticed.\n\n3. `too_large` is guarded by `not want`, so a rules.txt carrying both a real\n   coverage line and the marker compares against the line rather than falling\n   back to the refusal policy. A measurement outranks a policy; nothing pinned\n   that.\n\nThree selftest cases, each mutation-verified.\n\nThe fourth surviving mutation is not a defect: the rules.txt scan is\ncase-sensitive while the card scan is not, which is the right asymmetry --\nstrict about this module's own output, liberal about what the model writes.\nCommented rather than tested, so it does not read as an oversight.\n\n129 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: zufayu <8138883+zufayu@users.noreply.github.com>\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-11-18T04:41:15Z",
          "url": "https://github.com/XiaobingSuper/ATOM/commit/ec17e6f8df7b54d47e9e951651265ab049af39ae"
        },
        "date": 1795223429029,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 3030.12,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=545422 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.31,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=545422 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=545422 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.64,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=545422 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=545422 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 9771.47,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1758864 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.6,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1758864 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 2.73,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1758864 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 3.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1758864 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1758864 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 5094.46,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=917003 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.37,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=917003 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=917003 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=917003 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=917003 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 7967.98,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1434236 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1434236 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1434236 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.09,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1434236 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1434236 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 9571.89,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1722940 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.8,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1722940 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.4,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1722940 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 1.91,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1722940 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1722940 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 3094.61,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=557029 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.31,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=557029 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.35,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=557029 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.67,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=557029 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=557029 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 10335.13,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1860324 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1860324 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 2.58,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1860324 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 3.3,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1860324 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1860324 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 5177.78,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=932000 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=932000 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=932000 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=932000 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=932000 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 8169.28,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1470471 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1470471 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.74,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1470471 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 0.98,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1470471 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1470471 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 10094.58,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1817024 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1817024 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.32,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1817024 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 1.77,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1817024 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1817024 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 3114.41,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=560594 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.3,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=560594 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.35,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=560594 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=560594 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=560594 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 10371.49,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1866868 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1866868 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 2.58,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1866868 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 3.37,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1866868 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1866868 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 5209.66,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=937738 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=937738 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=937738 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.57,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=937738 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=937738 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 8022.29,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1444012 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1444012 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1444012 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.08,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1444012 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1444012 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 9659.88,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1738778 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.79,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1738778 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.39,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1738778 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 1.84,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1738778 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1738778 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44839863677"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zufayu",
            "username": "zufayu",
            "email": "zufa.yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ec17e6f8df7b54d47e9e951651265ab049af39ae",
          "message": "review-pr: publish the coverage verdict, and fail the gate when it cannot be checked (#3456)\n\n* review-pr: publish the coverage verdict, not just compute it\n\ntriage.py measures whether a diff fits one review pass, SKILL.md requires the\ncard to carry it, and _check_coverage_line gates that the card does. All three\nhave been in place since the port. No published review has ever carried the\nline.\n\nThe gate reads card.md. build_review assembles the body that is posted, and\nits prefix list named only \"Review:\" and \"Validation\". So the measurement was\ntaken, required, checked, and then dropped on the way out -- with every gate\ngreen, because they were watching the other copy.\n\nTwenty-one reviews went out this way. The one that mattered most was a\n10179-line, 54-file diff: it got two findings and no statement that two\nfindings on that much code are a sample rather than a sweep. _check_coverage_line's\nown docstring names this reader -- \"a reader cannot tell a card that omitted the\nline from one whose diff was small enough not to need it\" -- and that is exactly\nwho has been reading these.\n\nThe test fixture gains a Coverage line (asserting against an input that lacks\none proves nothing) and the new test pins both that it is published and that it\nsits above the findings index. Dropping \"Coverage:\" from the prefix list again\nturns it red.\n\ncheck_prompts.py 0 drift; selftest.sh 115 green; pytest 45 passed; black and\nruff clean.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): the coverage gate returned green when it read nothing\n\n`_check_coverage_line` compared the card's `Coverage:` line against the one\nStep 1 wrote into rules.txt. Three ways of reading nothing all returned `[]`,\nwhich the card gate cannot tell apart from a clean comparison:\n\n  - rules.txt missing or unreadable (`except OSError: return []`)\n  - rules.txt present but not Step 1's output (no `coverage:` line)\n  - `[diff-too-large]` -- GitHub refusing to serve a diff over 20000 lines\n\nThe third is the expensive one. That marker means the largest PR this bot can\never see, the exact case where \"the findings below are a sample, not a sweep\"\nhas to reach the reader -- and the card could omit coverage entirely and pass.\nA check that does not run looked exactly like a check that passed.\n\nNow: an unreadable or non-Step-1 rules.txt raises UNCHECKED-COVERAGE, and\n`[diff-too-large]` requires a coverage line on the card. `[underivable]` and\n`[derivation-failed]` keep returning `[]` and are named as the only entitled\nsilence: there Step 1 measured nothing, so no verdict exists for the card to\ndrop.\n\nSix selftest cases, four of them mutation-verified -- silencing any of the\nthree new paths, or widening the entitled-silence check to swallow\n`[diff-too-large]`, turns tests red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): say when a review could anchor nothing\n\nThe summary names the inline count only when there is one. With no anchors at\nall the block vanishes, so the review goes out with no inline comments and\nnothing saying any were expected.\n\nATOM#2152 -- 4546 lines over 8 files, the largest diff this bot has reviewed --\npublished six findings that way. Five cite no `path:line`; the sixth cites\nmerge-target files (`moe.py`, `loader.py`) the diff never touches, so\n`resolve_path` finds nothing to attach to. Six real findings, zero anchors,\nzero signal that the inline half of the review did not happen.\n\nThe surviving branch already argues this case against itself: \"an invisible\nfinding reads exactly like a finding that was never made.\" It stops one\ncondition short of its own reasoning -- partial invisibility is called out,\ntotal invisibility is not.\n\nNow a review with findings and no anchors says `0 of N finding(s) could be\nanchored to a line in this diff`, and says what anchoring needs. Mutation-\nverified: silencing the branch turns the test red. Dry-run against #2152's\nreal card emits exactly that line. 42 passed, 121 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* style: black the new anchor test\n\npsf/black@stable collapses the two-line .replace() call added with the\n0-anchor test. Formatting only.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): a diff GitHub refused to serve may not be called ordinary\n\nReview feedback on the previous commit: the `[diff-too-large]` branch checked\nonly that SOME coverage line was present, never what it said. So the softening\nthe measured path exists to refuse -- reporting a BEYOND ONE PASS diff as an\nordinary one -- walked straight through on the one diff large enough that\nGitHub would not serve it at all.\n\nOver 20000 lines is an order of magnitude past BIG_ADDED, so such a diff is\nBEYOND ONE PASS by definition and the card has to say the words.\n\nThe same review asked what the card should say when rules.txt has no line to\ncopy. Both messages now spell the line out verbatim instead of describing it.\n\nSeventh selftest case; mutation-verified (restoring the presence-only check\nturns it red). 122 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): match the coverage line the way the gate does, and drop two redundant reds\n\nReview feedback on 9113fd2 (all three verified before acting).\n\n1. The publisher reintroduced this PR's own bug one case over. The gate accepts\n   any line whose `strip().lower()` starts `coverage:`; `_line_starting` required\n   the exact prefix `Coverage:` at column 0. A card copying rules.txt's\n   `  coverage: ...` verbatim -- which SKILL.md's template asks for -- passed the\n   gate and lost the line on the way out. The new `[diff-too-large]` message made\n   it certain by spelling the hint in lower case. `_line_starting` now strips and\n   folds case, so gate and publisher accept the same line; the hint is written in\n   the card's own `Coverage:` form. Three new publisher cases (indented, lower\n   case, both).\n\n2. The two UNCHECKED-COVERAGE branches were a second red for a state already\n   blocked: the ledger gate (SKILL.md:380) opens rules.txt with no guard and exits\n   1 on LEDGER UNUSABLE, both before the card gate. Dropped, and the\n   `[diff-too-large]` case folded into the single card-line scan instead of a\n   second copy of the loop -- 72 lines to 44. Their selftest cases are replaced by\n   two that assert the ledger gate's exit code directly, because deleting a check\n   is only safe while the check replacing it is the one being measured.\n\n3. EMPTY-COVERAGE had no case at all: removing the branch left every test green.\n   Added one.\n\nComments at each site trimmed to what the code guarantees; the incident history\nlives in the commit log.\n\n124 green / 0 red, 43 publisher tests. Mutation-verified: restoring the strict\nprefix, dropping the `[diff-too-large]` branch, dropping the softening check, or\ndropping the empty check each turns tests red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: assert the anchor count, and finish the comment trim\n\nReview feedback on d177c3f (both items verified first).\n\n1. The anchor tests did not pin what they check. `test_a_review_that_anchors_nothing_says_so`\n   matched only the phrase, so dropping the count stayed green; nothing asserted the\n   line is ABSENT when some findings do anchor, so `elif loose:` degrading to\n   `if loose:` -- which emits the index block and the 0-of-N line together -- stayed\n   green too. Confirmed by running both mutations. Now the first asserts\n   `0 of 2 finding(s) could be anchored` (two of the fixture's findings fail to anchor,\n   not one) and the mixed-anchor test asserts the line is absent. Four mutations now\n   turn tests red: `elif`->`if`, dropping the count, silencing the branch, and\n   restoring `_line_starting`'s strict prefix.\n\n2. The previous commit message said the comments were trimmed; only triage.py was.\n   Trimmed the rest: _publish.py's coverage and 0-anchor blocks, selftest.sh's two\n   headers, and three test docstrings. Each now says what the code guarantees.\n\nAlso noted in review and confirmed here: fetch.sh:53-64 exits 1 when GitHub refuses a\ndiff over 20000 lines, so `[diff-too-large]` never reaches rules.txt in the run_one\npipeline. The branch is a backstop for a hand-run triage, not a live path; the PR\ndescription no longer calls it \"the surviving case\".\n\n124 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: call the diff-too-large rule a policy, and finish the trim\n\nReview feedback on f984656 (verified before acting).\n\nThe `[diff-too-large]` comment claimed a 20000-line refusal is \"an order of\nmagnitude past BIG_ADDED, so BEYOND ONE PASS by definition\". That does not\nfollow, and the review is right: GitHub's cap counts every line of the diff\n(fetch.sh), while BIG_ADDED counts only non-test ADDED lines (triage.py:1236-38,\nBIG_ADDED = 500). A deletion-heavy or test-heavy refusal could measure as one\npass. The comment now says BEYOND is required by policy, and gives the reason\nthat does hold: the diff was never served, so a review of what could not be read\nis a sample whatever the counts would have said. Behaviour is unchanged.\n\nLast two history sites trimmed: `_line_starting`'s docstring and the\n`[diff-too-large]` selftest header.\n\n124 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: say policy in the message too, and describe anchoring exactly\n\nReview feedback on c119eb7 (both verified).\n\n1. The previous commit made the `[diff-too-large]` COMMENT say \"policy, not\n   measurement\" and left the SOFTENED-COVERAGE MESSAGE opening \"Step 1 measured\n   this diff as BEYOND ONE PASS\" -- in the one case where no measurement exists.\n   The opening clause now branches on `too_large`: a refused diff gets \"GitHub\n   would not serve this diff, so it counts as BEYOND ONE PASS\", a measured one\n   keeps its wording. Two selftest cases pin both halves, mutation-verified in\n   both directions.\n\n2. The 0-of-N notice said a finding anchors when it cites a line \"the PR\n   actually changed\". `commentable()` records added AND context lines inside a\n   hunk, so \"inside this diff's hunks\" is what is true.\n\n126 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: drop a stray f-prefix and put the spelled line last\n\nCosmetic, from review. The first literal of the `why` expression carried an\n`f` with no placeholder, and the refused-diff message ended with the spelled\ncoverage line and then repeated \"sample\" after it. The suffix now comes last,\nso the message reads reason, consequence, then what to write.\n\nNo behaviour change. 126 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: cover three branches a mutation sweep found untested\n\nNot from review -- from mutating every logical line this PR adds and checking\nwhich mutations stayed green. Four did. Three were real gaps:\n\n1. The MISSING-COVERAGE reason could claim a measurement that does not exist.\n   This is the same defect review caught in SOFTENED-COVERAGE one commit ago,\n   and the fix there covered only that branch: swapping MISSING's `too_large`\n   reason for \"Step 1 measured the diff\" kept every test green.\n\n2. MISSING-COVERAGE's text field falls back to `[diff-too-large]` when there is\n   no `coverage:` line to name. Dropping the fallback printed an empty field and\n   nothing noticed.\n\n3. `too_large` is guarded by `not want`, so a rules.txt carrying both a real\n   coverage line and the marker compares against the line rather than falling\n   back to the refusal policy. A measurement outranks a policy; nothing pinned\n   that.\n\nThree selftest cases, each mutation-verified.\n\nThe fourth surviving mutation is not a defect: the rules.txt scan is\ncase-sensitive while the card scan is not, which is the right asymmetry --\nstrict about this module's own output, liberal about what the model writes.\nCommented rather than tested, so it does not read as an oversight.\n\n129 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: zufayu <8138883+zufayu@users.noreply.github.com>\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-11-18T04:41:15Z",
          "url": "https://github.com/XiaobingSuper/ATOM/commit/ec17e6f8df7b54d47e9e951651265ab049af39ae"
        },
        "date": 1795306226293,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 3148.17,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=566671 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.3,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=566671 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=566671 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.37,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=566671 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=566671 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 10395.22,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1871140 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 1.5,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1871140 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 2.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1871140 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 3.29,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1871140 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1871140 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 5248.82,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944787 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944787 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944787 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.52,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944787 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=944787 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 8276.43,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1489757 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.46,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1489757 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1489757 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 0.99,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1489757 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1489757 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 9918.81,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1785385 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1785385 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.35,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1785385 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 1.82,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1785385 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1785385 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 3152.78,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=567501 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.3,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=567501 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=567501 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.44,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=567501 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=567501 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 10339.68,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1861143 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 1.51,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1861143 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 2.58,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1861143 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 3.31,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1861143 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1861143 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 5241.36,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=943444 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=943444 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=943444 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=943444 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=943444 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 8302.76,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1494496 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1494496 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1494496 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 0.96,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1494496 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1494496 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 10003.84,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1800692 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1800692 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.34,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1800692 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 1.76,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1800692 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1800692 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 3120.94,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=561769 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.3,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=561769 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.34,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=561769 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.39,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=561769 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=561769 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 10316.88,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1857039 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 1.51,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1857039 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 2.59,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1857039 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 3.32,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1857039 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1857039 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 5238.29,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=942892 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.36,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=942892 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=942892 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=942892 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=942892 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 8291.68,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1492503 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.45,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1492503 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.73,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1492503 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 0.97,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1492503 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1492503 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 9970.11,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1794620 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1794620 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.34,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1794620 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 1.79,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1794620 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1794620 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/44959293131"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zufayu",
            "username": "zufayu",
            "email": "zufa.yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ec17e6f8df7b54d47e9e951651265ab049af39ae",
          "message": "review-pr: publish the coverage verdict, and fail the gate when it cannot be checked (#3456)\n\n* review-pr: publish the coverage verdict, not just compute it\n\ntriage.py measures whether a diff fits one review pass, SKILL.md requires the\ncard to carry it, and _check_coverage_line gates that the card does. All three\nhave been in place since the port. No published review has ever carried the\nline.\n\nThe gate reads card.md. build_review assembles the body that is posted, and\nits prefix list named only \"Review:\" and \"Validation\". So the measurement was\ntaken, required, checked, and then dropped on the way out -- with every gate\ngreen, because they were watching the other copy.\n\nTwenty-one reviews went out this way. The one that mattered most was a\n10179-line, 54-file diff: it got two findings and no statement that two\nfindings on that much code are a sample rather than a sweep. _check_coverage_line's\nown docstring names this reader -- \"a reader cannot tell a card that omitted the\nline from one whose diff was small enough not to need it\" -- and that is exactly\nwho has been reading these.\n\nThe test fixture gains a Coverage line (asserting against an input that lacks\none proves nothing) and the new test pins both that it is published and that it\nsits above the findings index. Dropping \"Coverage:\" from the prefix list again\nturns it red.\n\ncheck_prompts.py 0 drift; selftest.sh 115 green; pytest 45 passed; black and\nruff clean.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): the coverage gate returned green when it read nothing\n\n`_check_coverage_line` compared the card's `Coverage:` line against the one\nStep 1 wrote into rules.txt. Three ways of reading nothing all returned `[]`,\nwhich the card gate cannot tell apart from a clean comparison:\n\n  - rules.txt missing or unreadable (`except OSError: return []`)\n  - rules.txt present but not Step 1's output (no `coverage:` line)\n  - `[diff-too-large]` -- GitHub refusing to serve a diff over 20000 lines\n\nThe third is the expensive one. That marker means the largest PR this bot can\never see, the exact case where \"the findings below are a sample, not a sweep\"\nhas to reach the reader -- and the card could omit coverage entirely and pass.\nA check that does not run looked exactly like a check that passed.\n\nNow: an unreadable or non-Step-1 rules.txt raises UNCHECKED-COVERAGE, and\n`[diff-too-large]` requires a coverage line on the card. `[underivable]` and\n`[derivation-failed]` keep returning `[]` and are named as the only entitled\nsilence: there Step 1 measured nothing, so no verdict exists for the card to\ndrop.\n\nSix selftest cases, four of them mutation-verified -- silencing any of the\nthree new paths, or widening the entitled-silence check to swallow\n`[diff-too-large]`, turns tests red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): say when a review could anchor nothing\n\nThe summary names the inline count only when there is one. With no anchors at\nall the block vanishes, so the review goes out with no inline comments and\nnothing saying any were expected.\n\nATOM#2152 -- 4546 lines over 8 files, the largest diff this bot has reviewed --\npublished six findings that way. Five cite no `path:line`; the sixth cites\nmerge-target files (`moe.py`, `loader.py`) the diff never touches, so\n`resolve_path` finds nothing to attach to. Six real findings, zero anchors,\nzero signal that the inline half of the review did not happen.\n\nThe surviving branch already argues this case against itself: \"an invisible\nfinding reads exactly like a finding that was never made.\" It stops one\ncondition short of its own reasoning -- partial invisibility is called out,\ntotal invisibility is not.\n\nNow a review with findings and no anchors says `0 of N finding(s) could be\nanchored to a line in this diff`, and says what anchoring needs. Mutation-\nverified: silencing the branch turns the test red. Dry-run against #2152's\nreal card emits exactly that line. 42 passed, 121 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* style: black the new anchor test\n\npsf/black@stable collapses the two-line .replace() call added with the\n0-anchor test. Formatting only.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): a diff GitHub refused to serve may not be called ordinary\n\nReview feedback on the previous commit: the `[diff-too-large]` branch checked\nonly that SOME coverage line was present, never what it said. So the softening\nthe measured path exists to refuse -- reporting a BEYOND ONE PASS diff as an\nordinary one -- walked straight through on the one diff large enough that\nGitHub would not serve it at all.\n\nOver 20000 lines is an order of magnitude past BIG_ADDED, so such a diff is\nBEYOND ONE PASS by definition and the card has to say the words.\n\nThe same review asked what the card should say when rules.txt has no line to\ncopy. Both messages now spell the line out verbatim instead of describing it.\n\nSeventh selftest case; mutation-verified (restoring the presence-only check\nturns it red). 122 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): match the coverage line the way the gate does, and drop two redundant reds\n\nReview feedback on 9113fd2 (all three verified before acting).\n\n1. The publisher reintroduced this PR's own bug one case over. The gate accepts\n   any line whose `strip().lower()` starts `coverage:`; `_line_starting` required\n   the exact prefix `Coverage:` at column 0. A card copying rules.txt's\n   `  coverage: ...` verbatim -- which SKILL.md's template asks for -- passed the\n   gate and lost the line on the way out. The new `[diff-too-large]` message made\n   it certain by spelling the hint in lower case. `_line_starting` now strips and\n   folds case, so gate and publisher accept the same line; the hint is written in\n   the card's own `Coverage:` form. Three new publisher cases (indented, lower\n   case, both).\n\n2. The two UNCHECKED-COVERAGE branches were a second red for a state already\n   blocked: the ledger gate (SKILL.md:380) opens rules.txt with no guard and exits\n   1 on LEDGER UNUSABLE, both before the card gate. Dropped, and the\n   `[diff-too-large]` case folded into the single card-line scan instead of a\n   second copy of the loop -- 72 lines to 44. Their selftest cases are replaced by\n   two that assert the ledger gate's exit code directly, because deleting a check\n   is only safe while the check replacing it is the one being measured.\n\n3. EMPTY-COVERAGE had no case at all: removing the branch left every test green.\n   Added one.\n\nComments at each site trimmed to what the code guarantees; the incident history\nlives in the commit log.\n\n124 green / 0 red, 43 publisher tests. Mutation-verified: restoring the strict\nprefix, dropping the `[diff-too-large]` branch, dropping the softening check, or\ndropping the empty check each turns tests red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: assert the anchor count, and finish the comment trim\n\nReview feedback on d177c3f (both items verified first).\n\n1. The anchor tests did not pin what they check. `test_a_review_that_anchors_nothing_says_so`\n   matched only the phrase, so dropping the count stayed green; nothing asserted the\n   line is ABSENT when some findings do anchor, so `elif loose:` degrading to\n   `if loose:` -- which emits the index block and the 0-of-N line together -- stayed\n   green too. Confirmed by running both mutations. Now the first asserts\n   `0 of 2 finding(s) could be anchored` (two of the fixture's findings fail to anchor,\n   not one) and the mixed-anchor test asserts the line is absent. Four mutations now\n   turn tests red: `elif`->`if`, dropping the count, silencing the branch, and\n   restoring `_line_starting`'s strict prefix.\n\n2. The previous commit message said the comments were trimmed; only triage.py was.\n   Trimmed the rest: _publish.py's coverage and 0-anchor blocks, selftest.sh's two\n   headers, and three test docstrings. Each now says what the code guarantees.\n\nAlso noted in review and confirmed here: fetch.sh:53-64 exits 1 when GitHub refuses a\ndiff over 20000 lines, so `[diff-too-large]` never reaches rules.txt in the run_one\npipeline. The branch is a backstop for a hand-run triage, not a live path; the PR\ndescription no longer calls it \"the surviving case\".\n\n124 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: call the diff-too-large rule a policy, and finish the trim\n\nReview feedback on f984656 (verified before acting).\n\nThe `[diff-too-large]` comment claimed a 20000-line refusal is \"an order of\nmagnitude past BIG_ADDED, so BEYOND ONE PASS by definition\". That does not\nfollow, and the review is right: GitHub's cap counts every line of the diff\n(fetch.sh), while BIG_ADDED counts only non-test ADDED lines (triage.py:1236-38,\nBIG_ADDED = 500). A deletion-heavy or test-heavy refusal could measure as one\npass. The comment now says BEYOND is required by policy, and gives the reason\nthat does hold: the diff was never served, so a review of what could not be read\nis a sample whatever the counts would have said. Behaviour is unchanged.\n\nLast two history sites trimmed: `_line_starting`'s docstring and the\n`[diff-too-large]` selftest header.\n\n124 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: say policy in the message too, and describe anchoring exactly\n\nReview feedback on c119eb7 (both verified).\n\n1. The previous commit made the `[diff-too-large]` COMMENT say \"policy, not\n   measurement\" and left the SOFTENED-COVERAGE MESSAGE opening \"Step 1 measured\n   this diff as BEYOND ONE PASS\" -- in the one case where no measurement exists.\n   The opening clause now branches on `too_large`: a refused diff gets \"GitHub\n   would not serve this diff, so it counts as BEYOND ONE PASS\", a measured one\n   keeps its wording. Two selftest cases pin both halves, mutation-verified in\n   both directions.\n\n2. The 0-of-N notice said a finding anchors when it cites a line \"the PR\n   actually changed\". `commentable()` records added AND context lines inside a\n   hunk, so \"inside this diff's hunks\" is what is true.\n\n126 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: drop a stray f-prefix and put the spelled line last\n\nCosmetic, from review. The first literal of the `why` expression carried an\n`f` with no placeholder, and the refused-diff message ended with the spelled\ncoverage line and then repeated \"sample\" after it. The suffix now comes last,\nso the message reads reason, consequence, then what to write.\n\nNo behaviour change. 126 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: cover three branches a mutation sweep found untested\n\nNot from review -- from mutating every logical line this PR adds and checking\nwhich mutations stayed green. Four did. Three were real gaps:\n\n1. The MISSING-COVERAGE reason could claim a measurement that does not exist.\n   This is the same defect review caught in SOFTENED-COVERAGE one commit ago,\n   and the fix there covered only that branch: swapping MISSING's `too_large`\n   reason for \"Step 1 measured the diff\" kept every test green.\n\n2. MISSING-COVERAGE's text field falls back to `[diff-too-large]` when there is\n   no `coverage:` line to name. Dropping the fallback printed an empty field and\n   nothing noticed.\n\n3. `too_large` is guarded by `not want`, so a rules.txt carrying both a real\n   coverage line and the marker compares against the line rather than falling\n   back to the refusal policy. A measurement outranks a policy; nothing pinned\n   that.\n\nThree selftest cases, each mutation-verified.\n\nThe fourth surviving mutation is not a defect: the rules.txt scan is\ncase-sensitive while the card scan is not, which is the right asymmetry --\nstrict about this module's own output, liberal about what the model writes.\nCommented rather than tested, so it does not read as an oversight.\n\n129 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: zufayu <8138883+zufayu@users.noreply.github.com>\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-11-18T04:41:15Z",
          "url": "https://github.com/XiaobingSuper/ATOM/commit/ec17e6f8df7b54d47e9e951651265ab049af39ae"
        },
        "date": 1795394944302,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 4974.2,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=895356 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.19,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=895356 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.25,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=895356 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.55,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=895356 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=895356 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 16391.44,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2950459 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 0.95,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2950459 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 1.67,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2950459 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 2.96,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2950459 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2950459 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 7987.14,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1437686 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.23,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1437686 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.31,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1437686 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.44,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1437686 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1437686 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 12951,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2331180 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.29,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2331180 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2331180 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 0.95,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2331180 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2331180 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 16071.94,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2892949 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2892949 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 0.86,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2892949 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 1.58,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2892949 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2892949 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 4993.16,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=898769 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.19,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=898769 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.25,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=898769 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=898769 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=898769 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 16549.17,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2978851 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 0.94,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2978851 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 1.66,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2978851 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 2.84,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2978851 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2978851 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 8003.54,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1440637 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.23,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1440637 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.31,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1440637 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.4,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1440637 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1440637 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 12642.89,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2275720 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.29,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2275720 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2275720 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 0.97,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2275720 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2275720 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 15248.16,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2744669 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 0.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2744669 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 0.92,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2744669 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 1.96,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2744669 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2744669 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 5005.32,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=900957 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.19,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=900957 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.24,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=900957 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=900957 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=900957 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 16488.57,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2967943 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 0.95,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2967943 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 1.66,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2967943 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 2.77,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2967943 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=2967943 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 7958.43,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1432517 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.23,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1432517 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.32,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1432517 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1432517 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=1432517 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 12195.87,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2195257 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.3,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2195257 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 0.53,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2195257 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.07,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2195257 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=2195257 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 15838.49,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2850929 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 0.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2850929 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 0.87,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2850929 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 1.93,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2850929 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=2850929 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45083407887"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "zufayu",
            "username": "zufayu",
            "email": "zufa.yu@amd.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ec17e6f8df7b54d47e9e951651265ab049af39ae",
          "message": "review-pr: publish the coverage verdict, and fail the gate when it cannot be checked (#3456)\n\n* review-pr: publish the coverage verdict, not just compute it\n\ntriage.py measures whether a diff fits one review pass, SKILL.md requires the\ncard to carry it, and _check_coverage_line gates that the card does. All three\nhave been in place since the port. No published review has ever carried the\nline.\n\nThe gate reads card.md. build_review assembles the body that is posted, and\nits prefix list named only \"Review:\" and \"Validation\". So the measurement was\ntaken, required, checked, and then dropped on the way out -- with every gate\ngreen, because they were watching the other copy.\n\nTwenty-one reviews went out this way. The one that mattered most was a\n10179-line, 54-file diff: it got two findings and no statement that two\nfindings on that much code are a sample rather than a sweep. _check_coverage_line's\nown docstring names this reader -- \"a reader cannot tell a card that omitted the\nline from one whose diff was small enough not to need it\" -- and that is exactly\nwho has been reading these.\n\nThe test fixture gains a Coverage line (asserting against an input that lacks\none proves nothing) and the new test pins both that it is published and that it\nsits above the findings index. Dropping \"Coverage:\" from the prefix list again\nturns it red.\n\ncheck_prompts.py 0 drift; selftest.sh 115 green; pytest 45 passed; black and\nruff clean.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): the coverage gate returned green when it read nothing\n\n`_check_coverage_line` compared the card's `Coverage:` line against the one\nStep 1 wrote into rules.txt. Three ways of reading nothing all returned `[]`,\nwhich the card gate cannot tell apart from a clean comparison:\n\n  - rules.txt missing or unreadable (`except OSError: return []`)\n  - rules.txt present but not Step 1's output (no `coverage:` line)\n  - `[diff-too-large]` -- GitHub refusing to serve a diff over 20000 lines\n\nThe third is the expensive one. That marker means the largest PR this bot can\never see, the exact case where \"the findings below are a sample, not a sweep\"\nhas to reach the reader -- and the card could omit coverage entirely and pass.\nA check that does not run looked exactly like a check that passed.\n\nNow: an unreadable or non-Step-1 rules.txt raises UNCHECKED-COVERAGE, and\n`[diff-too-large]` requires a coverage line on the card. `[underivable]` and\n`[derivation-failed]` keep returning `[]` and are named as the only entitled\nsilence: there Step 1 measured nothing, so no verdict exists for the card to\ndrop.\n\nSix selftest cases, four of them mutation-verified -- silencing any of the\nthree new paths, or widening the entitled-silence check to swallow\n`[diff-too-large]`, turns tests red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): say when a review could anchor nothing\n\nThe summary names the inline count only when there is one. With no anchors at\nall the block vanishes, so the review goes out with no inline comments and\nnothing saying any were expected.\n\nATOM#2152 -- 4546 lines over 8 files, the largest diff this bot has reviewed --\npublished six findings that way. Five cite no `path:line`; the sixth cites\nmerge-target files (`moe.py`, `loader.py`) the diff never touches, so\n`resolve_path` finds nothing to attach to. Six real findings, zero anchors,\nzero signal that the inline half of the review did not happen.\n\nThe surviving branch already argues this case against itself: \"an invisible\nfinding reads exactly like a finding that was never made.\" It stops one\ncondition short of its own reasoning -- partial invisibility is called out,\ntotal invisibility is not.\n\nNow a review with findings and no anchors says `0 of N finding(s) could be\nanchored to a line in this diff`, and says what anchoring needs. Mutation-\nverified: silencing the branch turns the test red. Dry-run against #2152's\nreal card emits exactly that line. 42 passed, 121 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* style: black the new anchor test\n\npsf/black@stable collapses the two-line .replace() call added with the\n0-anchor test. Formatting only.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): a diff GitHub refused to serve may not be called ordinary\n\nReview feedback on the previous commit: the `[diff-too-large]` branch checked\nonly that SOME coverage line was present, never what it said. So the softening\nthe measured path exists to refuse -- reporting a BEYOND ONE PASS diff as an\nordinary one -- walked straight through on the one diff large enough that\nGitHub would not serve it at all.\n\nOver 20000 lines is an order of magnitude past BIG_ADDED, so such a diff is\nBEYOND ONE PASS by definition and the card has to say the words.\n\nThe same review asked what the card should say when rules.txt has no line to\ncopy. Both messages now spell the line out verbatim instead of describing it.\n\nSeventh selftest case; mutation-verified (restoring the presence-only check\nturns it red). 122 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* fix(review-pr): match the coverage line the way the gate does, and drop two redundant reds\n\nReview feedback on 9113fd2 (all three verified before acting).\n\n1. The publisher reintroduced this PR's own bug one case over. The gate accepts\n   any line whose `strip().lower()` starts `coverage:`; `_line_starting` required\n   the exact prefix `Coverage:` at column 0. A card copying rules.txt's\n   `  coverage: ...` verbatim -- which SKILL.md's template asks for -- passed the\n   gate and lost the line on the way out. The new `[diff-too-large]` message made\n   it certain by spelling the hint in lower case. `_line_starting` now strips and\n   folds case, so gate and publisher accept the same line; the hint is written in\n   the card's own `Coverage:` form. Three new publisher cases (indented, lower\n   case, both).\n\n2. The two UNCHECKED-COVERAGE branches were a second red for a state already\n   blocked: the ledger gate (SKILL.md:380) opens rules.txt with no guard and exits\n   1 on LEDGER UNUSABLE, both before the card gate. Dropped, and the\n   `[diff-too-large]` case folded into the single card-line scan instead of a\n   second copy of the loop -- 72 lines to 44. Their selftest cases are replaced by\n   two that assert the ledger gate's exit code directly, because deleting a check\n   is only safe while the check replacing it is the one being measured.\n\n3. EMPTY-COVERAGE had no case at all: removing the branch left every test green.\n   Added one.\n\nComments at each site trimmed to what the code guarantees; the incident history\nlives in the commit log.\n\n124 green / 0 red, 43 publisher tests. Mutation-verified: restoring the strict\nprefix, dropping the `[diff-too-large]` branch, dropping the softening check, or\ndropping the empty check each turns tests red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: assert the anchor count, and finish the comment trim\n\nReview feedback on d177c3f (both items verified first).\n\n1. The anchor tests did not pin what they check. `test_a_review_that_anchors_nothing_says_so`\n   matched only the phrase, so dropping the count stayed green; nothing asserted the\n   line is ABSENT when some findings do anchor, so `elif loose:` degrading to\n   `if loose:` -- which emits the index block and the 0-of-N line together -- stayed\n   green too. Confirmed by running both mutations. Now the first asserts\n   `0 of 2 finding(s) could be anchored` (two of the fixture's findings fail to anchor,\n   not one) and the mixed-anchor test asserts the line is absent. Four mutations now\n   turn tests red: `elif`->`if`, dropping the count, silencing the branch, and\n   restoring `_line_starting`'s strict prefix.\n\n2. The previous commit message said the comments were trimmed; only triage.py was.\n   Trimmed the rest: _publish.py's coverage and 0-anchor blocks, selftest.sh's two\n   headers, and three test docstrings. Each now says what the code guarantees.\n\nAlso noted in review and confirmed here: fetch.sh:53-64 exits 1 when GitHub refuses a\ndiff over 20000 lines, so `[diff-too-large]` never reaches rules.txt in the run_one\npipeline. The branch is a backstop for a hand-run triage, not a live path; the PR\ndescription no longer calls it \"the surviving case\".\n\n124 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: call the diff-too-large rule a policy, and finish the trim\n\nReview feedback on f984656 (verified before acting).\n\nThe `[diff-too-large]` comment claimed a 20000-line refusal is \"an order of\nmagnitude past BIG_ADDED, so BEYOND ONE PASS by definition\". That does not\nfollow, and the review is right: GitHub's cap counts every line of the diff\n(fetch.sh), while BIG_ADDED counts only non-test ADDED lines (triage.py:1236-38,\nBIG_ADDED = 500). A deletion-heavy or test-heavy refusal could measure as one\npass. The comment now says BEYOND is required by policy, and gives the reason\nthat does hold: the diff was never served, so a review of what could not be read\nis a sample whatever the counts would have said. Behaviour is unchanged.\n\nLast two history sites trimmed: `_line_starting`'s docstring and the\n`[diff-too-large]` selftest header.\n\n124 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: say policy in the message too, and describe anchoring exactly\n\nReview feedback on c119eb7 (both verified).\n\n1. The previous commit made the `[diff-too-large]` COMMENT say \"policy, not\n   measurement\" and left the SOFTENED-COVERAGE MESSAGE opening \"Step 1 measured\n   this diff as BEYOND ONE PASS\" -- in the one case where no measurement exists.\n   The opening clause now branches on `too_large`: a refused diff gets \"GitHub\n   would not serve this diff, so it counts as BEYOND ONE PASS\", a measured one\n   keeps its wording. Two selftest cases pin both halves, mutation-verified in\n   both directions.\n\n2. The 0-of-N notice said a finding anchors when it cites a line \"the PR\n   actually changed\". `commentable()` records added AND context lines inside a\n   hunk, so \"inside this diff's hunks\" is what is true.\n\n126 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: drop a stray f-prefix and put the spelled line last\n\nCosmetic, from review. The first literal of the `why` expression carried an\n`f` with no placeholder, and the refused-diff message ended with the spelled\ncoverage line and then repeated \"sample\" after it. The suffix now comes last,\nso the message reads reason, consequence, then what to write.\n\nNo behaviour change. 126 green / 0 red.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n* review-pr: cover three branches a mutation sweep found untested\n\nNot from review -- from mutating every logical line this PR adds and checking\nwhich mutations stayed green. Four did. Three were real gaps:\n\n1. The MISSING-COVERAGE reason could claim a measurement that does not exist.\n   This is the same defect review caught in SOFTENED-COVERAGE one commit ago,\n   and the fix there covered only that branch: swapping MISSING's `too_large`\n   reason for \"Step 1 measured the diff\" kept every test green.\n\n2. MISSING-COVERAGE's text field falls back to `[diff-too-large]` when there is\n   no `coverage:` line to name. Dropping the fallback printed an empty field and\n   nothing noticed.\n\n3. `too_large` is guarded by `not want`, so a rules.txt carrying both a real\n   coverage line and the marker compares against the line rather than falling\n   back to the refusal policy. A measurement outranks a policy; nothing pinned\n   that.\n\nThree selftest cases, each mutation-verified.\n\nThe fourth surviving mutation is not a defect: the rules.txt scan is\ncase-sensitive while the card scan is not, which is the right asymmetry --\nstrict about this module's own output, liberal about what the model writes.\nCommented rather than tested, so it does not read as an oversight.\n\n129 green / 0 red, 43 publisher tests.\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: zufayu <8138883+zufayu@users.noreply.github.com>\nCo-authored-by: Claude Opus 5 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-11-18T04:41:15Z",
          "url": "https://github.com/XiaobingSuper/ATOM/commit/ec17e6f8df7b54d47e9e951651265ab049af39ae"
        },
        "date": 1795497343449,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 request throughput",
            "value": 2335.41,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420373 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420373 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420373 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 p999 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420373 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc1 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420373 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 request throughput",
            "value": 7178.03,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1292045 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 avg latency",
            "value": 2.19,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1292045 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p99 latency",
            "value": 3.68,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1292045 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 p999 latency",
            "value": 4.56,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1292045 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc16 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1292045 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 request throughput",
            "value": 3888.12,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699861 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699861 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699861 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 p999 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699861 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc2 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=699861 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 request throughput",
            "value": 5809.86,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045774 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045774 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p99 latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045774 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 p999 latency",
            "value": 1.41,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045774 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc4 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1045774 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 request throughput",
            "value": 6910.37,
            "unit": "req/s",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1243866 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 avg latency",
            "value": 1.12,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1243866 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p99 latency",
            "value": 1.94,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1243866 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 p999 latency",
            "value": 2.53,
            "unit": "ms",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1243866 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-1p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-1p1d-conc8 router=pd policy=round_robin workers=2 prefill=1 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1243866 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 request throughput",
            "value": 2333.92,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420105 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420105 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420105 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 p999 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420105 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc1 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420105 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 request throughput",
            "value": 7159.73,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1288751 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 avg latency",
            "value": 2.19,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1288751 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p99 latency",
            "value": 3.69,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1288751 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 p999 latency",
            "value": 4.6,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1288751 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc16 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1288751 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 request throughput",
            "value": 3874.26,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=697367 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=697367 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=697367 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 p999 latency",
            "value": 0.77,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=697367 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc2 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=697367 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 request throughput",
            "value": 5863.59,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1055446 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1055446 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p99 latency",
            "value": 1.04,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1055446 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 p999 latency",
            "value": 1.37,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1055446 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc4 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1055446 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 request throughput",
            "value": 6921.16,
            "unit": "req/s",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1245808 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 avg latency",
            "value": 1.11,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1245808 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p99 latency",
            "value": 1.94,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1245808 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 p999 latency",
            "value": 2.5,
            "unit": "ms",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1245808 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-2p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-2p1d-conc8 router=pd policy=round_robin workers=3 prefill=2 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1245808 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 request throughput",
            "value": 2335.08,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420314 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 avg latency",
            "value": 0.41,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420314 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p99 latency",
            "value": 0.47,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420314 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 p999 latency",
            "value": 0.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420314 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc1 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc1 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=1 duration_seconds=180 request_number=420314 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 request throughput",
            "value": 7195.09,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1295116 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 avg latency",
            "value": 2.18,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1295116 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p99 latency",
            "value": 3.67,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1295116 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 p999 latency",
            "value": 4.54,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1295116 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc16 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc16 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=16 duration_seconds=180 request_number=1295116 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 request throughput",
            "value": 3877.38,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=697929 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 avg latency",
            "value": 0.49,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=697929 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p99 latency",
            "value": 0.63,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=697929 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 p999 latency",
            "value": 0.76,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=697929 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc2 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc2 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=2 duration_seconds=180 request_number=697929 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 request throughput",
            "value": 5828.66,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1049158 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 avg latency",
            "value": 0.65,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1049158 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p99 latency",
            "value": 1.05,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1049158 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 p999 latency",
            "value": 1.39,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1049158 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc4 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc4 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=4 duration_seconds=180 request_number=1049158 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 request throughput",
            "value": 6903.78,
            "unit": "req/s",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1242680 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 avg latency",
            "value": 1.12,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1242680 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p99 latency",
            "value": 1.94,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1242680 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 p999 latency",
            "value": 2.48,
            "unit": "ms",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1242680 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          },
          {
            "name": "Atomesh-Mocker::pd-chat-3p1d-conc8 failed requests",
            "value": 0,
            "unit": "count",
            "extra": "cell=pd-chat-3p1d-conc8 router=pd policy=round_robin workers=4 prefill=3 decode=1 producers=1 consumers=8 duration_seconds=180 request_number=1242680 Run: https://github.com/XiaobingSuper/ATOM/actions/runs/45322662353"
          }
        ]
      }
    ]
  }
}