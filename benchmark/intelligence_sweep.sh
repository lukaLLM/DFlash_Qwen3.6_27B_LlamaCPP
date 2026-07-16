#!/usr/bin/env bash
# Intelligence check (aiperf accuracy): gsm8k + math_500, greedy, sequential.
# Verified on aiperf 0.11.0. gsm8k+math_500 download only ~7 MB (one-time);
# lcb_codegeneration (LiveCodeBench, built into aiperf) is ~150-300 MB and
# grades pass@1 by running the generated code locally.
# Usage:  ./intelligence_sweep.sh dflash    # then:  mtp, then:  base
#         N=5 ./intelligence_sweep.sh dflash   # quick smoke test
#         BENCHES=lcb_codegeneration N=1 ./intelligence_sweep.sh dflash   # coding, 1 problem
#         STOP=1 ./intelligence_sweep.sh dflash   # stop the docker service when done
set -euo pipefail

case "${1:-dflash}" in
  dflash) svc=llamacpp_dflash;      tag=dflash; port=8001 ;;
  mtp)    svc=llama_cpp_qwen36_mtp; tag=mtp;    port=8001 ;;
  base)   svc=llamacpp_baseline;    tag=base;   port=8000 ;;
  *) echo "usage: $0 [dflash|mtp|base]"; exit 1 ;;
esac
# Per-benchmark default problem count. Set N=... to override for ALL benches.
# lcb_codegeneration emits up to 32k tokens/problem and runs the generated code
# locally to grade, so its default is deliberately tiny.
declare -A NDEF=( [gsm8k]=200 [math_500]=200 [lcb_codegeneration]=20 )
# BENCHES override, e.g. BENCHES=math_500 for the reasoning track
# (gsm8k caps answers at 256 tokens -> meaningless with --reasoning on).
# Add lcb_codegeneration for the coding track, e.g. BENCHES=lcb_codegeneration.
read -ra benches <<< "${BENCHES:-gsm8k math_500}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
art="$here/../artifacts/$tag/accuracy"

cd "$here/../docker"
docker compose up -d "$svc"
# STOP=1 -> tear the service down on exit (even if a benchmark fails).
[[ "${STOP:-0}" == 1 ]] && trap 'echo ">>> stopping $svc"; docker compose stop "$svc"' EXIT
until curl -sf localhost:$port/health >/dev/null; do sleep 2; done

# Aliases change as the compose file is tweaked -> always ask the live server.
model=$(curl -s localhost:$port/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
echo ">>> $svc  alias=$model"

for bench in "${benches[@]}"; do
  n="${N:-${NDEF[$bench]:-100}}"     # explicit N wins; else per-bench default
  echo ">>> $bench  problems=$n"
  aiperf profile \
    --model "$model" --url http://localhost:$port \
    --endpoint-type chat --streaming \
    --tokenizer Qwen/Qwen3.6-27B \
    --accuracy-benchmark "$bench" \
    --num-dataset-entries "$n" --dataset-sampling-strategy sequential \
    --request-count "$n" \
    --concurrency 1 --random-seed 42 \
    --extra-inputs temperature:0 --extra-inputs top_p:1.0 --extra-inputs top_k:1 \
    --output-artifact-dir "$art/$bench"
done

echo
echo "=== pass@1 ($tag, alias $model) ==="
grep -H OVERALL "$art"/*/accuracy_results.csv
echo "done -> $art   (stop the server before benchmarking the other one:"
echo "                docker compose stop $svc   -- or rerun with STOP=1)"
