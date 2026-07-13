#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Model downloader for the local compare-lab setup.
#
# Default downloads in this repo:
# - unsloth/DeepSeek-V4-Flash-GGUF (UD-IQ4_XS shards only)
#
# The script reads HF_TOKEN from the environment or a local .env file.
# Re-running resumes partial downloads automatically. A stall watchdog
# auto-restarts the download if no bytes arrive for STALL_SECS (default 300s).
# -----------------------------------------------------------------------------
# To Run
# source .venv/bin/activate
# chmod +x scripts/download_models.sh
# ./scripts/download_models.sh
set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f ".env" ]] && grep -qE '^[[:space:]]*(export[[:space:]]+)?HF_TOKEN=' ".env"; then
    # Load HF_TOKEN from local project env file if present.
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
  fi
fi

# Always prefer this repo's venv, even if another venv is active.
# (Running from e.g. the LiveCodeBench venv picks up an 'hf' without
# hf_xet and the download crawls/stalls on the legacy HTTPS fallback.)
if [[ -x ".venv/bin/hf" ]]; then
  export PATH="$PWD/.venv/bin:$PATH"
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing dependency: 'hf' command not found."
  echo "Install with uv:"
  echo "  uv init"
  echo "  uv venv .venv"
  echo "  source .venv/bin/activate"
  echo "  uv add huggingface-hub hf-xet"
  echo "Then activate your venv and run this script again."
  exit 127
fi

# Xet-backed repos (e.g. unsloth GGUFs) stall or crawl at ~150 kB/s without
# the hf_xet package. Refuse to start a 100+ GB download without it.
hf_py="$(dirname "$(command -v hf)")/python3"
if ! "$hf_py" -c "import hf_xet" >/dev/null 2>&1; then
  echo "Missing dependency: 'hf_xet' is not installed for $(command -v hf)."
  echo "Without it, downloads use the slow legacy path and stall. Install it:"
  echo "  uv add hf-xet    (or: uv pip install -p .venv hf-xet)"
  exit 127
fi

# Xet backend tuning. High-performance mode opens many parallel connections;
# on this line it repeatedly STALLED transfers (0 B/s forever), so it is
# opt-in now:  XET_HIGH_PERF=1 ./scripts/download_models.sh
# If default Xet mode still stalls, bypass Xet entirely (slower but plain
# HTTPS):      HF_HUB_DISABLE_XET=1 ./scripts/download_models.sh
if [[ "${XET_HIGH_PERF:-0}" == 1 ]]; then
  export HF_XET_HIGH_PERFORMANCE=1
fi
# Avoid deprecation warning if this variable exists in the user shell.
unset HF_HUB_ENABLE_HF_TRANSFER || true
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN found (from env or .env). Using token for downloads."
else
  echo "HF_TOKEN not found in environment or .env. Continuing without login."
  echo "For private/gated repos, set HF_TOKEN in .env or run:"
  echo "export HF_TOKEN=\"your_hf_token_here\""
fi

# ---------------------------------------------------------------------------
# Stall watchdog. hf/Xet downloads sometimes hang forever mid-transfer
# (0 B/s for hours, no error). Poll the repo's cache size; if it stops
# growing for STALL_SECS, kill the download and restart it - hf resumes
# from the cache, so already-downloaded bytes are never lost.
# Tune with:  STALL_SECS=120 MAX_RETRIES=50 ./scripts/download_models.sh
# ---------------------------------------------------------------------------
STALL_SECS="${STALL_SECS:-300}"
MAX_RETRIES="${MAX_RETRIES:-30}"
POLL_SECS=15

repo_cache_bytes() {
  local dir="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${1//\//--}"
  local bytes
  bytes="$(du -sb "$dir" 2>/dev/null | cut -f1)"
  echo "${bytes:-0}"
}

download_or_exit() {
  local repo="$1"
  shift

  local attempt pid rc last_bytes now_bytes stalled
  for (( attempt=1; attempt<=MAX_RETRIES; attempt++ )); do
    hf download "$repo" --type model "$@" &
    pid=$!
    trap 'kill "$pid" 2>/dev/null; echo; echo "Interrupted by Ctrl+C."; exit 130' INT TERM

    last_bytes="$(repo_cache_bytes "$repo")"
    stalled=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep "$POLL_SECS"
      now_bytes="$(repo_cache_bytes "$repo")"
      if [[ "$now_bytes" == "$last_bytes" ]]; then
        stalled=$(( stalled + POLL_SECS ))
        if (( stalled >= STALL_SECS )); then
          echo
          echo "No progress for ${STALL_SECS}s -> restarting download (attempt $attempt/$MAX_RETRIES)."
          kill "$pid" 2>/dev/null
          wait "$pid" 2>/dev/null || true
          continue 2
        fi
      else
        stalled=0
        last_bytes="$now_bytes"
      fi
    done

    rc=0
    wait "$pid" || rc=$?
    trap - INT TERM
    if (( rc == 0 )); then
      return 0
    fi
    if (( rc == 130 )); then
      echo "Interrupted by Ctrl+C."
      exit 130
    fi
    echo "hf download failed (exit $rc); retrying in 5s (attempt $attempt/$MAX_RETRIES)."
    sleep 5
  done

  echo "Failed: $repo after $MAX_RETRIES attempts."
  exit 1
}

GGUF_MODELS=(
  "unsloth/DeepSeek-V4-Flash-GGUF:UD-IQ4_XS"
)

HF_MODELS=(
)

for SPEC in "${GGUF_MODELS[@]}"; do
  REPO="${SPEC%%:*}"
  QUANT="${SPEC#*:}"
  if [[ "$REPO" == "$QUANT" ]]; then
    QUANT="Q4_K_M"
  fi

  echo "Downloading ${QUANT} for $REPO..."
  download_or_exit "$REPO" --include "*${QUANT}*.gguf"

  echo "Done: $REPO (${QUANT})"
  echo "----------------------------------------"
done

if [[ ${#HF_MODELS[@]} -eq 0 ]] && [[ ${#GGUF_MODELS[@]} -eq 0 ]]; then
  echo "No models selected. Edit download_models.sh and add entries to HF_MODELS."
  exit 0
fi

for REPO in "${HF_MODELS[@]}"; do
  echo "Downloading full repo for $REPO..."
  download_or_exit "$REPO"

  echo "Done: $REPO"
  echo "----------------------------------------"
done
