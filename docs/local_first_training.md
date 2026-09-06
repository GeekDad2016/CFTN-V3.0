# Local-first specialist training

## Current evidence and starting point

The latest local V12 checkpoint is epoch 150, at `C:/CFTN/artifacts/math_master_experiment_v12/run/checkpoint_epoch_0150.pth`. The run stopped at its acceptance budget. Its saved active probes were linear equations 12/12, powers 6/12 and Pythagoras 8/12, with earlier-skill retention 63/76. These are small development probes, not general Maths certification.

V3 now includes a self-contained, state-dictionary-compatible port of that 7,172,865-parameter Maths tower and its 260-token byte tokenizer. A real RTX 4070 comparison against the legacy implementation gave identical full-forward logits. Neither GPT nor another specialist is constructed by the local trainer. Native requests remain typed Maths JSON, with the original `Problem: ...\nSolution:` framing. Targets are compact worked equations plus an answer.

## Local schedule

1. Audit source hashes, preserve semantic split assignments, and encode every example with the real tokenizer.
2. Measure the imported checkpoint, then run a disposable 32-example overfit test. Require at least 95% answer accuracy within 200 updates; discard those updates.
3. Train multiplication prerequisites, bounded powers, Pythagoras, fractions, percentages, division and linear equations sequentially. Each epoch visits the active examples without replacement and mixes approximately 25% criterion-balanced replay. Length bucketing reduces padding. Computed values and final answers receive greater loss weight than formatting tokens.
4. Each phase needs two consecutive checks at 95% answer accuracy and 90% exact worked-trace accuracy, with no measured retention decline. Then run a separate test panel. Stop after 30 epochs if the phase does not pass; do not promote a failed stage just because its budget ended.
5. Retest all configured phases before exporting `math.specialist`. Some panels, notably bounded powers, are small; passing means success within this development curriculum only. It is not broad Maths or PhD readiness. Original V12 data and checkpoints are preserved.
6. Train the next specialists independently, with native interfaces and capability-specific checks. String can reuse the historical specialist approach. General reasoning and Python need separately defined curricula; the existing narrow formal-logic exercises are not a general reasoner.
7. After specialists pass, assemble their weights into a single `.cftn` artifact and move coordination training to RunPod. Train request formulation, routing, gates and message integration on checked compositions, with frozen specialists initially. Native assembly does not activate the new bridges.

The current generated dataset has 1,302 multiplication and 107 bounded-power training examples, alongside preserved V12 families. Its maximum encoded length is 211 tokens, compared with the old 2,048-token capacity. Powers are limited to bases -20..20 and exponents 2..4. The original broader powers scope is deferred, not silently counted as mastered.

## Execution and learning are separate

The Dispatcher can select only active capabilities. Only selected towers execute, and actual calls are recorded in `last_execution_trace`. Tests count embedding forward calls: a code/logic plan executes those two towers once each and no others; a coordinator-only request executes zero specialists. UpdatePlan continues to determine which parameters may change independently of which towers execute.

Before RunPod coordination, add causal comparison panels (coordinator alone, prescribed route, predicted route, each tower removed, and donor-message replacements). The old single-example batch-shuffle hook is not a valid causal ablation. The legacy word-problem continual runner explicitly rejects bundles containing these typed native specialists.

## Commands and files

Start locally:

```powershell
./scripts/start_local_math.ps1
```

Artifacts: `G:/ctfn-text/artifacts/v3_local_math_v1`

Dataset: `G:/ctfn-text/data/v3_local_math_v1`

Dashboard: `http://192.168.1.128:8792`

To request a safe stop, create `STOP` in the artifact directory. The worker stops after its next saved epoch. To resume, remove STOP and use the start script after confirming the previous process has exited. The current specialist checkpoint contains weights, optimizer and RNG state. A stale lock must be inspected, not blindly removed.

Assemble an accepted specialist later, preserving the base artifact:

```text
python -m cftn_v3.local_specialist --base base.cftn --specialist math.specialist --output assembled.cftn
```

This stores the specialist architecture, tokenizer identity and weights in one bundle; new bridges require training before activation. Full-scale assembly is deferred until the native specialists are ready. The self-contained save/load path has been tested with a small coordinator bundle.
