#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/root/.cache/huggingface
CONFIG="${CFTN_CONFIG:-artifacts/profile/selected_config.json}"
test -f "$CONFIG" || { echo 'Run production profiling first, or set CFTN_CONFIG explicitly.'; exit 1; }
STEPS="${CFTN_STEPS:-1000}"
python -m cftn_v3.cli prepare
mkdir -p artifacts
if [ "${CFTN_RESUME_AFTER_ROUTING:-0}" = 1 ]; then
  test -f artifacts/bootstrap.cftn
else
  python -m cftn_v3.cli train --config "$CONFIG" --mode routing --steps "$STEPS" --output artifacts/bootstrap.cftn
fi
for tower in math string code formal_logic science retrieval long_context multilingual tool_use structured_data information_extraction commonsense; do
  python -m cftn_v3.cli train --bundle artifacts/bootstrap.cftn --tower "$tower" --steps "$STEPS" --output artifacts/bootstrap.cftn
done
python -m cftn_v3.cli train --bundle artifacts/bootstrap.cftn --mode communication --steps "$STEPS" --output artifacts/bootstrap.cftn
python -m cftn_v3.cli train --bundle artifacts/bootstrap.cftn --mode integration --steps "$STEPS" --output artifacts/bootstrap.cftn
python -m cftn_v3.cli evaluate --bundle artifacts/bootstrap.cftn --activate
