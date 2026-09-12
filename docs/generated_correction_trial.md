# Generated-answer checking with supervised correction

This is a bounded Stage 2 pilot, not RL and not an automatic replacement for every stage's trainer. The protected snapshot was taken at round 786, cursor 1. Both arms start from these identical weights, optimizer and RNG states and receive five fresh rounds of 4,096 examples. If the original start is retained, its partial-round cursor is preserved.

The verifier independently computes expected answers for supported Stage 2 operations and number comparison. It checks generated arithmetic equations and comparisons, records the first false step, and identifies wrong answers or malformed output. It accepts canonical procedures and direct operand-linked addition/subtraction alternatives. Other correct-answer traces are marked unverified and excluded from failure mining; this is not a general proof verifier.

Mining uses a deterministic criterion-balanced panel from training data only. Validation, test and quarantined comparison families are excluded, including side swaps and commutative addition. Gold targets are independently checked before mining. Each generation, error classification and correct target is saved in mined_generations.json. No wrong generated continuation is used as a training target.

Correction training samples 60% from verified generated failures, 20% from normal active-stage examples and 20% from earlier stages. The loss remains supervised cross-entropy with existing target weighting plus SIGReg 0.0005. Feedback identifies the failure for inspection; weight updates use the original verified gold target, not feedback prose or a reward. Failure mining occurs once in this initial controlled trial.

Full active and retention panels evaluate the protected start and both endpoints. Candidate adoption requires an improvement over normal training and no regression on any answer, trace or format error count against either baseline or start. Otherwise a non-regressing baseline or the original checkpoint is selected. The process then resumes the curriculum automatically. Stage promotion still requires the existing full gates; this pilot does not certify mastery or change evaluation targets.

Artifacts and logs: G:/ctfn-text/artifacts/v3_1/generated_correction_probe. The dashboard follows the active phase. A failure records error.json and leaves the protected original available. Results are pending at implementation time.
