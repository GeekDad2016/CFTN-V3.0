#!/usr/bin/env bash
# Source this file in every training, evaluation and dashboard shell.
export CFTN_PROJECT_ROOT=/workspace/CFTN-V3.2
export CFTN_ARTIFACT_ROOT=/workspace/cftn/artifacts
export CFTN_DATA_ROOT=/workspace/cftn/data
export CFTN_BUNDLE_SCRATCH=/tmp/cftn-bundles
export HF_HOME=/tmp/cftn-cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR=/tmp/cftn-cache/pip
export TORCH_HOME=/tmp/cftn-cache/torch
export XDG_CACHE_HOME=/tmp/cftn-cache
export WANDB_DIR=/workspace/cftn/logs/wandb
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PATH="/opt/cftn-v32-venv/bin:$PATH"
