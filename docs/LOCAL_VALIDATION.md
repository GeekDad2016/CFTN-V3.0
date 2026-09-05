# Local validation — 5 September 2026

Environment: Python from `C:/Users/adria/anaconda3/python.exe`, PyTorch
2.9.0+cu126, Transformers 4.56.2, safetensors 0.6.2, RTX 4070.

- 49 tests passed, including a real CUDA selective update and exact artifact
  round trip, deterministic optimizer resume, unchanged unrelated optimizer
  state, gated-message gradient flow, English/Romanian coverage, restricted
  Python expression execution, read-only SQLite, queue idempotence, and
  fail-closed retention/release checks.
- CLI bilingual fixture build audited 312 records across all twelve towers
  and a math-to-string composition fixture.
- CLI CUDA routing, communication, and integration one-step jobs completed
  and wrote self-contained `.cftn` candidate bundles.
- Tiny profiling instantiated all twelve towers and the tiny coordinator:
  759,764 parameters at the time of measurement, 21.7 MB peak allocated and
  27.3 MB reserved. This profile is not the production model-size selection.
- Local dashboard/API launched at `http://127.0.0.1:8790`.

The source subsequently includes a learned dependency head; the exact current
parameter count must be taken from a fresh profile. The recorded measurement
above is evidence for the earlier tiny integration smoke only.

RunPod endpoint supplied: `root@103.196.86.190:15961`. SSH attempts rejected the
public key while local work completed. Production Qwen loading, 80 GB profiling,
remote deployment, twelve trained capability gates, and accepted live updates
are not validated yet. No candidate was promoted by these smoke tests.
