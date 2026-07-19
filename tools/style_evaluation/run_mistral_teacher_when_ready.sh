#!/usr/bin/env bash
# Wait for an existing teacher_search PID, then run mistral Local bake-off.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
LOG=/tmp/teacher_search_mistral.log
SEARCH_PID="${1:-}"

if [[ -z "$SEARCH_PID" ]]; then
  SEARCH_PID="$(pgrep -f 'python3 tools/style_evaluation/teacher_search.py --limit 10 --context 32768 --require-local --models gemma-4-31b-styletune' | head -1 || true)"
fi

{
  if [[ -n "${SEARCH_PID}" ]] && kill -0 "$SEARCH_PID" 2>/dev/null; then
    echo "waiting for teacher_search pid=$SEARCH_PID at $(date -Is)"
    while kill -0 "$SEARCH_PID" 2>/dev/null; do sleep 10; done
    echo "prior search finished at $(date -Is)"
  else
    echo "no prior search running; starting mistral now at $(date -Is)"
  fi
  export LLM_DISABLE_THINKING=1
  exec python3 tools/style_evaluation/teacher_search.py \
    --limit 10 --context 32768 --require-local \
    --models mistral-nemo-style-step2000
} >>"$LOG" 2>&1
