# V3.2 RunPod workspace

Code: `/workspace/CFTN-V3.2`, branch `codex/v3.2` from
`https://github.com/GeekDad2016/CFTN-V3.0.git`.

The user allocates 50 GB of persistent network storage at `/workspace`.
The shared filesystem's `df` reports pool capacity, not the user's quota.
The current container root reports 20 GB; do not assume it provides 50 GB.

Persistent: source/Git history, curated/generated dataset splits and manifests,
checkpoints, optimizer/RNG state, validation evidence, logs and environment lock.
Use `/workspace/cftn/data`, `/workspace/cftn/artifacts` and `/workspace/cftn/logs`.
Do not keep downloaded raw-source caches or pretrained-model caches here.

Disposable: Python environment `/opt/cftn-v32-venv`, downloads and model caches
`/tmp/cftn-cache`, bundle staging `/tmp/cftn-bundles`. Rebuild these after migration.

```bash
cd /workspace/CFTN-V3.2
bash scripts/setup_runpod_v32.sh
source scripts/runpod_v32_env.sh
bash scripts/start_runpod_v32_dashboard.sh /workspace/cftn/artifacts/next_expert
```

The dashboard binds port 8789. Use the RunPod HTTP proxy for this port. It reads
saved evidence and exposes the existing queued-validation control; it does not
start training. The initial next-expert directory has no training results.

Environment variables above configure cache locations and provide conventional
paths. **math32 reads data/root from its JSON configuration directly**: a future
RunPod training config must explicitly use the persistent paths. Its existing
resume check also binds the data path string, so a Windows-checkpoint migration
needs a reviewed relocation change with dataset-hash verification first.

No training migration or new expert training is launched by setup. The active
local V3.2 run remains separate. Source includes local working-tree fixes for the
all-dataset phase label, validation-loss trigger (0.002), and dashboard history
and polling; these are part of this synchronized revision.
