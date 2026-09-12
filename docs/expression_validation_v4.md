# Stage 2 failure audit and validation expansion

The completed round 544 report contained nine failed examples across active and retention panels. Expected trace equations were correct. No failure was an equivalent mathematical trace rejected solely for formatting. Errors included omitted borrowing, `11+0=111`, and declaring 10 equal to 20. The scorer and strict first-five-stage 100% gates remain unchanged.

The validation-only derivative `G:/ctfn-text/data/v3_1_balanced_validation_v4` preserves every original question and adds 6 less-than and 263 equality expression comparisons. Each outcome now has 296 questions, with 1,018 total Stage 2 validation questions. Training and sealed test files are byte-identical to foundation_v3. Added questions exclude comparison families already present in train, validation, test or quarantined held-out records, including side swaps and commutative addition variants. Gold scoring and tokenizer length checks passed.

The checkpoint was backed up as `before_balanced_validation_round_547.specialist`. Round 546 training updates were already saved before its unfinished full evaluation was interrupted. Resume starts at round 547, preserving weights, optimizer, RNG, controller counters and cursor. Only the dataset manifest binding and pass streaks changed. The last completed dashboard report predates the expansion; new full evaluations use the larger panel, so aggregate scores are not directly comparable.

Reproduction tools: `scripts/audit_foundation_failures.py`, `cftn_v3/balance_expression_validation.py`, and the one-time stopped-worker migration `scripts/adopt_balanced_validation.py`. Detailed audit and migration JSON files reside in the maths artifact directory.
