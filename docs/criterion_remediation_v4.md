# Criterion-only remediation

The v4 local pipeline continues the v3 Maths checkpoint at Stage 2, preserving model weights, optimizer, RNG and completed-stage history. A new dataset/policy lineage resets the exhausted remediation budget. Inheritance requires the checkpoint dataset hash to match the new manifest's parent hash; subsequent resumes require the exact v4 dataset and policy.

Training follows this sequence:

1. A new stage gets three normal rounds, with 75% current-stage examples and 25% accepted-stage replay when prior stages exist.
2. Failed active criteria take priority over failed retention criteria. Repair selects one criterion deterministically and samples 100% from its training records. No other criterion or replay enters this repair batch.
3. Two consecutive passing focused checks lead to three normal-mixture consolidation rounds. Active and prior criteria remain evaluated throughout repair, so forgetting remains visible.
4. Promotion requires two consecutive passing normal checks, all three consolidation rounds after repair, two full active validation and cumulative retention passes on consecutive training rounds. Regressions select another focused repair. Exhaustion blocks progression; it never skips a stage.

The active policy now has one limit: 240 total rounds per stage, including consolidation. There is no repair-attempt cutoff. Full active validation and full prior-stage retention run immediately after three normal consolidation rounds, and on the immediately following round after a full pass. The 20-round schedule is a fallback during prolonged recovery. Promotion requires consecutive full passes plus the normal consolidation requirement. A final check also runs at the stage ceiling. Routine panels guide remediation but cannot promote a stage. The v1 attempt-blocked checkpoint is backed up and migrated explicitly, preserving round 47 and optimizer/RNG; the 240-round budget is not reset. Each round draws 4,096 examples with replacement; it is not a full pass over every dataset record. The learning rate remains 0.00001. The prepared String tower starts only after Maths passes final acceptance and releases the GPU.

The sampler balances criterion, operation, decision outcome and procedural case. Comparing expressions includes all four addition/subtraction combinations and equal, adjacent and separated results. Non-decision tasks are not balanced by every distinct numerical answer. Format and worked-trace checks remain required, including tasks whose correct final answer is always true.

## Dataset audit

Maths v4 has 192,773 training, 6,807 validation and 6,804 test records. It preserves the old training set and adds 2,304 comparison records, evenly split across `<`, `=` and `>`. Added training excludes held-out reversed and commuted comparison families, including quarantined records. All gold responses were checked with the procedural scorer and actual tokenizer; maximum length is 641 of 2,048 tokens.

Equality was approximately 1% of the original elementary and place-value comparison training pools. Those pools are now balanced during sampling. Raw dataset proportions remain visible in `balance_audit.json`; majority examples were not discarded. Twenty-four unseen elementary comparison examples were added to each held-out split. Validation panels balance available decision classes without replacement, and each decision class must separately meet the answer/format thresholds (and active trace threshold), so overall accuracy cannot hide equality failures. Small held-out class counts still limit statistical confidence.

The audit also found mixed-operation imbalance in counting and arithmetic. Operation-balanced sampling addresses their effective training distribution. Constant-true Euclid/identity contracts retain their mathematically correct labels; answer accuracy alone is insufficient evidence of learning them. The advanced curriculum remains selected procedural mathematics, not comprehensive graduate mathematics.

String v3 preserves 99,955 training, 9,888 validation and 9,862 test records, with the same balanced sampler and criterion-only controller. Its contains yes/no distribution was already balanced. Maximum length is 228 of 256 tokens.

## Running and inspecting

Run `scripts/start_local_curriculum.ps1`, which defaults to `config/local_curriculum_v4.json`. Data and artifacts reside on G: as configured. The dashboard on port 8792 should use `G:/ctfn-text/artifacts/v3_curriculum_v4`. Status exposes `training_mode`, `focused_criterion`, `batch_criteria`, `batch_decisions`, and consolidation progress. Checkpoint metadata includes the repair controller and batch cursor. A STOP file in the active tower output directory requests a checkpoint at the next completed update.

Validation: unit/controller and runner tests cover class balancing, held-out exclusion, repair-only batches, three-round consolidation, regression, attempt limits, dashboard gates and checkpoint state. A disposable RTX 4070 update and checkpoint reload succeeded. `scripts/validate_criterion_cuda.py` reproduces that bounded CUDA check without changing the live checkpoint.
