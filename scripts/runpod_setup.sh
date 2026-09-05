#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv --system-site-packages
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pip freeze > environment.lock.txt
python -m pytest -q
python -m cftn_v3.cli profile --device cuda --root artifacts/profile
