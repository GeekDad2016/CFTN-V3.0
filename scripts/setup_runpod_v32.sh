#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runpod_v32_env.sh
mkdir -p "$CFTN_ARTIFACT_ROOT" "$CFTN_DATA_ROOT" /workspace/cftn/logs "$CFTN_BUNDLE_SCRATCH"
python3 -m venv /opt/cftn-v32-venv
python -m pip install --no-cache-dir --upgrade pip
# Blackwell requires a newer CUDA/PyTorch build than this pod image provides.
python -m pip install --no-cache-dir 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-cache-dir -e '.[test]'
python -m pip freeze > /workspace/cftn/logs/environment.lock.txt
printf 'Ready. Activate with: source /workspace/CFTN-V3.2/scripts/runpod_v32_env.sh\n'
