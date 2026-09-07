# Full local curriculum repair

The earlier seven-topic repair run stopped without acceptance. This release uses its compatible saved weights, but resets curriculum acceptance and optimizer state because the supervision has changed. It revalidates and trains from stage zero. Future-stage knowledge in the inherited weights is not evidence of mastery under the new contract.

## Data

The immutable V11 source is `C:/CFTN/.datasets/math_master_experiment_v11_stage8_powers_signal_remediation_r2`. The derivative is `G:/ctfn-text/data/v3_full_math_v2`. All 141,130 source training records are preserved. Typed-input solvers reconstruct compact worked procedures and a single answer; tuples use brackets and commas. Source answer agreement is checked before writing. Repairs cover 11,071 training, 41 validation and 42 test records with duplicate answer blocks.

The derivative has 162,050 training, 3,533 validation and 3,459 test records. There are 5,313 duplicate training semantics retained for provenance; the sampler deduplicates within each criterion. All encoded examples were checked with the actual byte tokenizer: maximum 641 of 2,048 tokens. Source hashes and source semantic plus canonical-input split isolation are verified. New polynomial-identity counterfactual families share a split.

The added tasks include general rational quadratics, degree 2–4 polynomial differentiation/integration, rational binomial probabilities, 3×3 determinants, derivative evaluation, extended Euclid certificates, finite weighted expectations, series convergence at a point, permutation orders and nontriangular 2×2 spectra. The stage titles describe these bounded computations. This is **not** comprehensive graduate mathematics, arbitrary symbolic algebra, natural-language theorem proving or PhD capability. Held-out examples test unseen inputs within these structures; passing them does not establish unrestricted structural generalization.

## Training and acceptance

Run `scripts/start_local_curriculum.ps1`. It refuses duplicate local workers. A detached pipeline uses `config/local_curriculum_v2.json`. Data and artifacts live on G:. Only one native specialist is loaded, with no coordinator or other specialists running.

Each round samples 2,048 examples: 75% active-stage and 25% criterion-balanced replay from previously accepted stages. Stage zero uses only active data. Length bucketing limits padding. The learning rate is 5e-5. Eight normal rounds are followed, if necessary, by up to three six-round remediation attempts. Remediation draws from stage-specific, training-only short-procedure pools and full active problems; prior-skill failures also receive targeted replay. No validation/test questions enter the training batch.

Every round evaluates a fixed panel of up to 12 distinct examples per active criterion and four per prior criterion. Every active criterion requires 95% answer and format accuracy and 90% exact canonical worked-trace accuracy; prior answer/format performance must meet its retention threshold. Exact trace matching is deliberately separate from answer correctness and may reject alternative valid derivations. Two consecutive passes trigger complete active validation and a larger cumulative retention gate. With very small panels, percentage thresholds can require every answer correct.

Promotion only follows passing those checks. A bounded-round stop is `blocked`, never success. Final sealed tests are opened only after all stages, with up to 32 examples per criterion; a test failure is terminal for this release. Validation is repeatedly used for decisions, so it must not be represented as an unbiased final test. A consumed-test marker prevents automatic adaptive retries after interruption.

The current specialist checkpoint is saved each round and every 100 updates, with optimizer, Torch/CUDA RNG, deterministic batch cursor, curriculum hash and policy. Place a `STOP` file in the active tower artifact directory for a safe stop at the next training update. Remove it only when resuming intentionally. A new run initializes from the previous compatible weights; subsequent resumes restore optimizer and cursor. A blocked run requires inspection rather than an automatic reset of its retry budget.

## String handoff and assembly

`G:/ctfn-text/data/v3_string_v2` contains 99,955 training, 9,888 validation and 9,862 test examples for exact ASCII length, count, zero-based index, reverse, contains and substitution. Source strings shared with earlier splits are excluded from held-out sets, including 112 validation and 137 test records. All answers are independently recomputed using Python string operations. Inputs are typed requests; the compatible six-layer legacy String checkpoint is preserved as an initial native artifact. Maximum encoded length is 228 of 256 tokens.

The pipeline launches String only when the Maths process exits, the final artifact is accepted and bound to the configured dataset hash, and the GPU lock is released. A crash, pause, exhausted remediation budget or failed test blocks the handoff. Native Maths and String artifacts can be assembled into one `.cftn` file; newly installed towers remain inactive until their coordinator bridges and routing are trained and evaluated. Coordinator, gates and Dispatcher training remain a later RunPod task.

## Dashboard and evidence

Run the dashboard with `--root G:/ctfn-text/artifacts/v3_curriculum_v2 --host 0.0.0.0 --port 8792`. It follows the queue's active tower and shows stage scopes, dataset counts, rounds, evaluation progress, per-criterion metrics, remediation attempt, worker health, checkpoint, before/after samples and validation history. It marks a missing worker as an error. The API is read-only and does not load a model.

The CUDA smoke report is saved in `G:/ctfn-text/artifacts/v3_curriculum_v2/cuda_smoke.json`. A disposable six-example test performed 20 real RTX 4070 updates; loss fell from 0.6561 to 0.00456. This confirms optimization works, not held-out mastery. Original production weights were not changed by that test.
