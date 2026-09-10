# V3.1 local maths training revision

Fresh 9.32M-parameter maths tower (+29.98%), integrated remediation and early place-value scaffolds. Current run: `config/local_curriculum_v31.json`. See [V3.1 training details](docs/v3_1_training.md). V3.0 checkpoints remain preserved separately.

# CFTN V3.0

### Automatic learning experiment

`python -m cftn_v3.learning_experiment --wait` waits for tower repairs and then
runs continuous rounds of Math, retrieval/facts, Python, logic, and Math-to-String
communication learning until paused from the dashboard. Each stage has 50 updates.
Finite datasets wrap around and cached teacher responses are reused; this does
not imply an unlimited supply of novel knowledge. Teacher responses are explicitly
unverified experimental supervision; the default training verifier and public
ingestion remain unchanged. Checkpoints in `artifacts/learning_experiment` never
activate a release. Reports compare held-out reference loss and sample exact-match
accuracy before/after; equivalent code can be undercounted by exact matching.
BoolQ (`google/boolq`, CC-BY-SA-3.0, revision
`35b264d03638db9f4ce671b711558bf7ff0f80d5`) provides passage-based factual questions,
not a current-world knowledge source. GSM8K and local synthetic questions cover
the other domains. Source revisions and licenses are recorded in the dataset
manifest. Completed stages persist in checkpoint metadata for restart safety.
This experiment tests learning behavior, not broad competence or deployability.
Each round also trains delegation: supervised Dispatcher tower/round/dependency
selection with routing replay, coordinator adapter subtask-request generation,
local specialist tasks, and differentiable bridge/final-answer synthesis. The
first multi-tower scope is Math-to-String and Math-to-Code, alongside single-tower
tasks. Automatic evaluation predicts the route and request texts, validates the
plan, and compares final accuracy against coordinator-only responses. This does
not establish general-purpose planning. Dashboard Automatic delegation tests
show selected towers and generated requests, with explicit fallback on invalid
plans. Teacher answer correctness and experimental deployment policy are unchanged.
The dashboard queues questions for checkpoint boundaries and compares optional
expected answers. Explicitly checked teaching examples can enter subsequent
matching-tower updates; unchecked tests never enter training. Pause preserves the
current block before taking effect. Checkpoints replace the experimental current
file atomically; the repaired starting checkpoint remains preserved. Each completed
stage saves a report and the dataset position in checkpoint metadata.

### Offline English teacher cycles

Install `.[teacher]`, then run `python -m cftn_v3.teacher_cycles prepare` to
download `openai/gsm8k` at revision `740312add88f781978c0658806c59bc2815b9866`
(MIT) into persistent `data/teacher`. The test split is excluded from teacher
generation and training; each cycle also reserves a disjoint new-material panel.
`python -m cftn_v3.teacher_cycles cycle` generates at most 128 Qwen answers and,
once enough verified records exist, runs at most 100 Math-only continual updates
with 25% replay. It requires an accepted bootstrap release and the shared GPU
lock. Repeat the command for subsequent bounded cycles; the database cursor
persists across restarts. Rejected answers never enter training. Only final
numeric answers are checked against the pinned reference; teacher reasoning is
not validated or used as supervision. This is a math expansion, not broad
all-domain distillation. Full release gates remain mandatory.

Subsequent training stages record a 32-example English development evaluation
per target tower, including sample predictions. These are informational and do
not replace the full release panels. Earlier already-running stages do not gain
this hook retroactively; communication/integration evaluate all towers.

RunPod storage: deploy this repository at `/workspace/V3.0`. Run the shell
scripts from this checkout; they change to the project directory before running.
Datasets (`data/`), checkpoints, logs and live state (`artifacts/`) therefore
remain on persistent `/workspace` storage. The private Git remote is at
`/workspace/V3.0.git`. Downloaded model assets use `/root/.cache/huggingface`
on the pod's local disk and can be downloaded again when replacing the pod.

Selective continual learning across twelve scratch-trained specialists, with
English and Romanian interfaces and a frozen pretrained coordinator.

This is a separate project. It does not modify or load V12 training state.
The first implementation provides executable model/training/release plumbing
and bounded bilingual fixtures. No specialist is marked accepted by default.
Passing unit tests is not a claim that twelve useful specialists have been trained.

## Quick local validation

Python 3.11+ with CUDA PyTorch:

```powershell
python -m pip install -e '.[test]'
python -m pytest -q
python -m cftn_v3.cli prepare --tiny --data artifacts/smoke_data
python -m cftn_v3.cli train --tiny --data artifacts/smoke_data --tower math --steps 2 --output artifacts/smoke.cftn
python -m cftn_v3.cli evaluate --bundle artifacts/smoke.cftn --data artifacts/smoke_data --tower math --limit 2
```

Tiny is a real twelve-tower integration test, not a deployable Qwen model.
The evaluation should fail competence gates on random or briefly trained weights.

## RunPod

Use an 80 GB-class or larger GPU. Supply your Git remote, exact commit, and SSH
endpoint; credentials are never committed. `scripts/deploy.ps1` clones the
independent repository into `/workspace/V3.0` and runs setup and profiling.

```bash
bash scripts/runpod_setup.sh
bash scripts/bootstrap.sh
```

Setup installs the environment, records `environment.lock.txt`, runs tests,
and profiles small/medium/large models. Profiling includes all twelve resident
towers, optimizer-state allocation, 4,096-token math and 16,384-token long-context
backward passes. The largest profile below 80% VRAM becomes
`artifacts/profile/selected_config.json`. Bootstrap consumes that measured file.
Production profiling refuses a GPU below the 80 GB class.

Coordinator/teacher: `Qwen/Qwen3-4B-Instruct-2507`, pinned revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. All twelve specialist embeddings and
layers initialize randomly; specialist parameters are never shared across towers.
Only the exact-string tower uses the independent UTF-8 byte tokenizer.

Bootstrap trains routing, each specialist, communication, then integration.
`CFTN_STEPS` controls each stage (default 1,000, maximum 10,000). It then runs
complete release gates. A failed model remains a candidate. More data/training
may be required; the script does not weaken gates or loop indefinitely.

## Training and communication

`ExecutionPlan` selects up to two towers per round for at most four rounds.
It validates capability availability and dependency order. `UpdatePlan`
separately authorizes parameter groups using verified supervision.

Ordinary specialist/continual updates select one tower. The coordinator,
dispatcher, all other towers, and bridges remain outside the optimizer.
Integration can update selected towers, their bridges, coordinator receivers,
and low-rank coordinator adapters. Routing updates are a separate mode.
Frozen parameters remain differentiable where necessary; they are not globally
wrapped in `no_grad`. Routing features use the frozen coordinator base only.

Request and return bridges exchange gated differentiable messages. Coordinator
receivers operate on the output workspace; they do not insert hooks into Qwen's
internal layers. Specialist receivers inject requests inside the tower blocks,
allowing later specialist computation to depend on the request. Teacher-forced
specialist targets are not fed back as messages.
Standalone auxiliary objectives retain specialist supervision during integration.
The dispatcher learns wake, round, and dependency predictions. Typed plans are
validated before execution; exact-string outputs retain their original bytes.

The baseline bootstrap fixtures teach tightly bounded operations, including
math addition, exact Unicode reversal, pure additive Python functions, finite
logic, units, document lookup, English/Romanian quantity translation, mock tool
calls, SQLite lookup, schema extraction, and explicit object persistence.
Math→string examples additionally exercise sequential cooperation.

These fixtures do not measure broad translation or commonsense ability. Their
held-out numeric IDs also exercise extrapolation. Importing broader curated
data and reviewing Romanian teacher outputs are necessary before broader claims.

```bash
python -m cftn_v3.cli teacher --input data/train.jsonl --output data/teacher_verified.jsonl --limit 100
```

Teacher generation is offline and accepts only verified outputs. The byte/string
student uses sequence distillation. No logit distillation is silently performed
across different vocabularies.

## Private live learning

First obtain an accepted release:

```bash
python -m cftn_v3.cli evaluate --bundle artifacts/bootstrap.cftn --activate
python -m cftn_v3.cli serve --host 127.0.0.1 --port 8790
python -m cftn_v3.cli learn --watch --steps 200
```

Run the serving and learner processes separately. A GPU lock serializes work.
Chat requests wait during finite training jobs. The serving process loads only
the accepted bundle. Baseline and candidate evaluation also run sequentially.
Use an SSH tunnel for the private dashboard and API; there is no public-user
authentication system in this single-user release.

- `GET /`: dashboard.
- `GET /api/status`: accepted release, candidate progress, PID liveness, status age.
- `POST /api/chat`: `{ "prompt": "...", "language": "ro" }` or `"en"`.
- `POST /api/feedback`: `{ "event_id": "...", "tower": "math", "target": "...", "confirm": true }`.
- `POST /api/ingest`: candidate record; user-supplied verification claims are ignored.

Private corrections can also be explicitly confirmed from the CLI:

```bash
python -m cftn_v3.cli ingest --input corrections.jsonl --confirm
python -m cftn_v3.cli learn --tower math --force --steps 200
```

Records need `prompt`, `target`, `tower`, and `language`. They may specify
`criterion`, `knowledge_kind` (`skill` or `fact`), `stable_fact`, and `supersedes`.
Changing facts are stored in retrievable memory and excluded from parameter
updates. Confirmed stable facts can train. Superseded records stop participating
in pending sampling. Retrieval is a deterministic lexical baseline.

A cycle becomes due at 128 verified examples, or after 24 hours with at least
32. Explicit flush also needs 32. Semantic groups are reserved for new-material
evaluation before training; at least eight held-outs are required. A separately
supplied `artifacts/new_panel_<tower>.jsonl` overrides automatic holdout selection.
Replays contain at most 4,096 stratified records, sampled as 25% of each batch.
English and Romanian replay coverage is retained when available.

The learner evaluates both languages, previous skills, integration, routing,
and held-out new material. Failures retain the previous accepted release and
leave the input records unconsumed. Successful activation and ingestion-cursor
advancement are one SQLite transaction.

## Acceptance and artifacts

Native gates require 500 examples per language per tower: 99% exact-string,
90% other bounded capabilities. English performance cannot hide Romanian
failures. Release also requires 99% routing-set accuracy, 99.5% valid routing,
zero inactive calls, and at most one percentage point of retention/integration
loss. Two fixed seeds are evaluated. Greedy decoding is deterministic; these
are reproducibility checks, not independent statistical samples.

The development panels select candidates; `data/test.jsonl` stays separate for
milestone checks. Human-confirmed open-ended corrections use exact confirmed
targets; that verifier does not certify arbitrary equivalent answers.

One `.cftn` ZIP bundle contains safetensors weights, tokenizers, architecture,
registry, and metadata. Resume bundles additionally contain optimizer and RNG
state, loaded through restricted `weights_only=True`. Export strips training
state. Private interaction logs and replay remain outside deployment bundles.
Checksums, strict tensor shapes, safe archive paths, and atomic replacement
protect loading and interrupted writes. The SQLite release ledger supports
rollback; accepted artifacts are never overwritten by candidate training.

```bash
python -m cftn_v3.cli export --bundle artifacts/bootstrap.cftn --output artifacts/deploy.cftn
python -m cftn_v3.cli rollback
```

## Remaining evidence to collect on RunPod

The local test suite proves control-path behavior and bounded CUDA execution.
Full pinned-Qwen loading, measured production profile choice, twelve trained
capability gates, quality of Romanian teacher outputs, and sustained live
learning require the configured remote GPU and actual training runs. None are
reported as completed merely because their commands and gates exist.
