#!/usr/bin/env bash
# Replay the SAME LiveCodeBench prompts (fixed order, greedy) via aiperf.
# A/B speed test: dflash vs dflash-ngram on real coding prompts.
# Usage:  ./lcb_speed.sh dflash 100    # then:  ./lcb_speed.sh ngram 100
#         ./lcb_speed.sh ngram 3       # smoke run
#         ./lcb_speed.sh ngram 100 path/to/other.inputs.json
# Dataset comes from lcb_to_aiperf.py; sampling params live IN the payloads
# (aiperf sends inputs-json verbatim - --extra-inputs would be ignored).
set -euo pipefail

case "${1:-}" in
  dflash) svc=llamacpp_dflash;       tag=dflash;       port=8001 ;;
  ngram)  svc=llamacpp_dflash_ngram; tag=dflash_ngram; port=8001 ;;
  mtp)    svc=llama_cpp_qwen36_mtp;  tag=mtp;          port=8001 ;;
  base)   svc=llamacpp_baseline;     tag=base;         port=8000 ;;
  *) echo "usage: $0 [dflash|ngram|mtp|base] [N] [inputs.json]"; exit 1 ;;
esac
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
art="$here/../artifacts/$tag/lcb_speed"
data="${3:-$here/data/lcb_release_v5_first100.inputs.json}"
[ -f "$data" ] || { echo "no dataset at $data - run lcb_to_aiperf.py first" >&2; exit 1; }

# Warmup entries are prepended in the file (aiperf's sequential sampler consumes
# them first); N counts only the real problems measured after warmup.
read -r WARM TOTAL <<<"$(python3 -c "
import json,sys
m=json.load(open('$data'))['manifest']
print(m['warmup_entries'], m['num_problems'])")"
N="${2:-$TOTAL}"
[ "$N" -le "$TOTAL" ] || { echo "N=$N > $TOTAL problems in $data (sampler would wrap)" >&2; exit 1; }

cd "$here/../docker"
# dflash/ngram/mtp all publish host 8001 - only one may run, or the health
# check below would silently measure whichever server already owns the port.
docker compose stop llamacpp_dflash llamacpp_dflash_ngram llama_cpp_qwen36_mtp >/dev/null 2>&1 || true
docker compose up -d "$svc"
for _ in $(seq 1 300); do curl -sf localhost:$port/health >/dev/null && break; sleep 2; done
curl -sf localhost:$port/health >/dev/null || { echo "$svc never became healthy on :$port" >&2; exit 1; }

# Aliases change as the compose file is tweaked -> always ask the live server.
model=$(curl -s localhost:$port/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
echo ">>> $svc  alias=$model  warmup=$WARM  measured=$N/$TOTAL"

# Payloads are sent verbatim, so patch the model field to the live alias.
# The patched copy stays in the artifact dir for traceability.
mkdir -p "$art"
python3 -c "
import json
d = json.load(open('$data'))
for e in d['data']:
    for p in e['payloads']:
        p['model'] = '$model'
json.dump(d, open('$art/inputs.patched.json', 'w'), indent=2, sort_keys=True)"

warmup_args=()
[ "$WARM" -gt 0 ] && warmup_args=(--warmup-request-count "$WARM")
aiperf profile \
  --model "$model" --url http://localhost:$port \
  --endpoint-type chat --streaming \
  --tokenizer Qwen/Qwen3.6-27B \
  --custom-dataset-type inputs_json --input-file "$art/inputs.patched.json" \
  --dataset-sampling-strategy sequential \
  --concurrency 1 --random-seed 42 \
  "${warmup_args[@]}" --request-count "$N" \
  --output-artifact-dir "$art"

# Draft acceptance is NOT on /metrics - read draft_n / draft_n_accepted from a
# completion's timings block (one real LCB prompt, same greedy settings).
python3 -c "
import json
d = json.load(open('$art/inputs.patched.json'))
p = dict(d['data'][int('$WARM')]['payloads'][0])
p['stream'] = False
p['timings_per_token'] = True
print(json.dumps(p))" \
  | curl -s localhost:$port/v1/chat/completions -H 'Content-Type: application/json' -d @- \
  | python3 -c "
import json,sys
t=json.load(sys.stdin)['timings']
d,a=t.get('draft_n',0),t.get('draft_n_accepted',0)
print(f\"lcb q0 draft {a}/{d} accepted ({100*a/d:.0f}%), {t['predicted_per_second']:.1f} tok/s\" if d else 'lcb q0 no draft stats (not a spec server?)')" \
  > "$art/acceptance.txt" || true
cat "$art/acceptance.txt" 2>/dev/null || true
echo "done -> $art"
