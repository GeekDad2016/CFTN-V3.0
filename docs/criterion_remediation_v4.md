# Stage-first Maths training

The active controller (version 3) gives each stage up to 240 normal-mixture rounds before automatic recovery. Normal batches contain 75% current-stage examples, balanced across criteria, and 25% earlier-stage replay. A full passing check triggers another full check after the next training round; two consecutive full passes allow earlier promotion. Routine success after at least three normal rounds can trigger a full check, but routine failure does not trigger early recovery. Full checks otherwise run every 20 total stage rounds.

If the normal block is exhausted and full validation still fails, one weak criterion receives a bounded recovery block: 80% target examples and 20% other current/prior skills. Recovery ends after two passing focused routine checks or 30 rounds, then starts another normal block. Promotion must occur on normal training, after at least three normal rounds. At most four recovery blocks per stage are allowed as an explicit safety budget, each separated by a full normal block. This is not the old early four-attempt cutoff.

The live Stage 6 checkpoint was migrated after round 60: three completed normal rounds and 57 completed historical recovery rounds. All model updates and optimizer/RNG were preserved. `before_stage_first.specialist` preserves the pre-migration checkpoint; `stage_first_migration.json` records the policy and counters. The new schedule resumes at total stage round 61 with normal block counter 3/240. Historical recovery is counted separately. Normal and recovery counts are visible on the dashboard. Dataset hashes and optimizer settings must match on migration; subsequent resumes require controller version 3 and an identical policy.

Each round draws 4,096 examples with replacement, rather than traversing the entire dataset. Learning rate remains 0.00001, dropout 0.1 and AdamW weight decay 0.01. No SIGReg was added. Data, held-out isolation, answer contracts and per-class acceptance thresholds remain unchanged. String starts only after Maths passes final acceptance.

## Dataset audit

Maths v4 has 192,773 training, 6,807 validation and 6,804 test records. It preserves the old training set and adds 2,304 comparison records, evenly split across `<`, `=` and `>`. Added training excludes held-out reversed and commuted comparison families, including quarantined records. All gold responses were checked with the procedural scorer and actual tokenizer; maximum length is 641 of 2,048 tokens.

Equality was approximately 1% of the original elementary and place-value comparison training pools. Those pools are now balanced during sampling. Raw dataset proportions remain visible in `balance_audit.json`; majority examples were not discarded. Twenty-four unseen elementary comparison examples were added to each held-out split. Validation panels balance available decision classes without replacement, and each decision class must separately meet the answer/format thresholds (and active trace threshold), so overall accuracy cannot hide equality failures. Small held-out class counts still limit statistical confidence.

The audit also found mixed-operation imbalance in counting and arithmetic. Operation-balanced sampling addresses their effective training distribution. Constant-true Euclid/identity contracts retain their mathematically correct labels; answer accuracy alone is insufficient evidence of learning them. The advanced curriculum remains selected procedural mathematics, not comprehensive graduate mathematics.

String v3 preserves 99,955 training, 9,888 validation and 9,862 test records, with the same balanced sampler and criterion-only controller. Its contains yes/no distribution was already balanced. Maximum length is 228 of 256 tokens.

## Running and inspecting

Run `scripts/start_local_curriculum.ps1`, which defaults to `config/local_curriculum_v4.json`. Data and artifacts reside on G: as configured. The dashboard on port 8792 should use `G:/ctfn-text/artifacts/v3_curriculum_v4`. Status exposes `training_mode`, `focused_criterion`, `batch_criteria`, `batch_decisions`, and consolidation progress. Checkpoint metadata includes the repair controller and batch cursor. A STOP file in the active tower output directory requests a checkpoint at the next completed update.

Validation: unit/controller and runner tests cover class balancing, held-out exclusion, repair-only batches, three-round consolidation, regression, attempt limits, dashboard gates and checkpoint state. A disposable RTX 4070 update and checkpoint reload succeeded. `scripts/validate_criterion_cuda.py` reproduces that bounded CUDA check without changing the live checkpoint.
