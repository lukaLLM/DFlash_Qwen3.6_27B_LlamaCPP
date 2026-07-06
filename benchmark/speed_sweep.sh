#!/usr/bin/env bash
# Synthetic speed sweep (no downloads). Input = output tokens, greedy, concurrency 1.
# Usage:  ./speed_sweep.sh dflash    # then:  mtp, then:  base
set -euo pipefail

case "${1:-dflash}" in
  dflash) svc=llamacpp_dflash;      tag=dflash; port=8001 ;;
  mtp)    svc=llama_cpp_qwen36_mtp; tag=mtp;    port=8001 ;;
  base)   svc=llamacpp_baseline;    tag=base;   port=8000 ;;
  *) echo "usage: $0 [dflash|mtp|base]"; exit 1 ;;
esac
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
art="$here/../artifacts/$tag/speed"

# size -> measured requests (fewer as size grows). Last one is the high-context point.
sizes=(512 4096 12288 36864)
declare -A REQS=( [512]=30 [4096]=10 [12288]=5 [36864]=3 )
declare -A WARM=( [512]=2  [4096]=2  [12288]=1 [36864]=1 )

# Optional: restrict the sweep to specific sizes (must be keys above).
#   ./speed_sweep.sh dflash 36864        # just the high-context point
#   ./speed_sweep.sh dflash 4096 12288   # a couple of mid points
if [ "$#" -gt 1 ]; then
  sizes=( "${@:2}" )
  for N in "${sizes[@]}"; do
    [ -n "${REQS[$N]:-}" ] || { echo "no REQS/WARM entry for size $N (known: ${!REQS[*]})" >&2; exit 1; }
  done
fi

cd "$here/../docker"
docker compose up -d "$svc"
until curl -sf localhost:$port/health >/dev/null; do sleep 2; done

# Aliases change as the compose file is tweaked -> always ask the live server.
model=$(curl -s localhost:$port/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
echo ">>> $svc  alias=$model"

for N in "${sizes[@]}"; do
  aiperf profile \
    --model "$model" --url http://localhost:$port \
    --endpoint-type chat --streaming \
    --tokenizer Qwen/Qwen3.6-27B \
    --synthetic-input-tokens-mean "$N" --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean          "$N" --output-tokens-stddev          0 \
    --extra-inputs temperature:0 --extra-inputs top_p:1.0 --extra-inputs top_k:1 \
    --extra-inputs ignore_eos:true --extra-inputs min_tokens:"$N" \
    --concurrency 1 --random-seed 42 \
    --warmup-request-count "${WARM[$N]}" --request-count "${REQS[$N]}" \
    --output-artifact-dir "$art/isl${N}_osl${N}"
done

# Draft acceptance is NOT on /metrics (llama.cpp exports no spec counters there).
# Read draft_n / draft_n_accepted from a completion's `timings` block per size.
# Fresh file on a full run; append when only a subset of sizes was requested.
[ "$#" -gt 1 ] || : > "$art/acceptance.txt"
for N in "${sizes[@]}"; do
  curl -s localhost:$port/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a very long story.\"}],\"max_tokens\":$N,\"temperature\":0,\"timings_per_token\":true}" \
    | python3 -c "
import json,sys
t=json.load(sys.stdin)['timings']
d,a=t.get('draft_n',0),t.get('draft_n_accepted',0)
print(f\"osl=$N draft {a}/{d} accepted ({100*a/d:.0f}%), {t['predicted_per_second']:.1f} tok/s\" if d else f\"osl=$N no draft stats (not a spec server?)\")"
done >> "$art/acceptance.txt" || true
echo "done -> $art"
