#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runpod_v32_env.sh
root="${1:-/workspace/cftn/artifacts/next_expert}"
case "$root" in /workspace/cftn/artifacts/*) ;; *) echo 'Dashboard artifact root must be under /workspace/cftn/artifacts/' >&2; exit 1;; esac
mkdir -p "$root"
if ss -ltnH 'sport = :8789' | grep -q .; then
  echo 'Port 8789 is already in use; existing service left running.' >&2
  exit 1
fi
nohup setsid python -u -m cftn_v3.local_math_dashboard --root "$root" --host 0.0.0.0 --port 8789 > "$root/dashboard.stdout.log" 2> "$root/dashboard.stderr.log" < /dev/null &
echo "$!" > "$root/dashboard.pid"
echo "Dashboard PID $(cat "$root/dashboard.pid"), artifact root $root"
