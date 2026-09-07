# Foundation expansion and longer training

This revision resumes from `G:/ctfn-text/artifacts/v3_curriculum_v2/math/current.specialist`. The previous run stopped at round 26 with 35/36 active answers correct; its three-number addition criterion had one persistent incorrect answer. It was not an accepted tower. The new run preserves these learned weights, initializes a fresh optimizer for the changed data and learning rate, and revalidates acceptance. The old checkpoint, dataset and reports remain on disk.

## Settings

`config/local_curriculum_v3.json` is now the default for `scripts/start_local_curriculum.ps1`.

| Setting | Previous | Revised |
| --- | ---: | ---: |
| Learning rate | 0.00005 | 0.00001 |
| Normal rounds per stage | 8 | 120 |
| Remediation attempts | 3 | 4 |
| Rounds per remediation attempt | 6 | 30 |
| Maximum rounds per stage | 26 | 240 |
| Examples per round | 2,048 | 4,096 |
| Quick validation examples per criterion | Up to 12 | Up to 32 |
| Retention examples per criterion | Up to 4 | Up to 8 |

These are sampled training rounds, not full dataset epochs. Accepted stages advance early; the maximum budget is a ceiling. The same mastery thresholds, two consecutive passes, full stage validation, cumulative retention and final sealed tests still apply. More rounds and a lower learning rate do not guarantee generalization. Previously observed validation failures are not inserted into training.

## Dataset

The derivative is `G:/ctfn-text/data/v3_full_math_v3`: 190,469 training, 6,783 validation and 6,780 test records.

| Stage | Previous training records | Revised | Supported scope |
| --- | ---: | ---: | --- |
| 1 | 3,908 | 5,245 | Counting, comparison and two/three-term sums up to 20 |
| 2 | 312 | 6,036 | Addition/subtraction within 20; missing minuends/subtrahends; comparing two arithmetic expressions |
| 3 | 378 | 8,887 | Two-digit place value; composing tens and ones; crossing tens; missing values and place-value comparisons |
| 4 | 3,462 | 3,462 | Addition/subtraction within 100 |
| 5 | 13,000 | 13,000 | Multiplication/division with factors up to 100 |
| 6 | 18,135 | 30,984 | Existing multiplication/division plus addition and nonnegative subtraction with three/four-digit operands up to 9,999 |

Finite tasks such as identifying tens and ones only have a small number of distinct inputs in their numeric range. Additional records add different tasks and combinations rather than claiming repeated copies are new knowledge. The dashboard also shows distinct task counts. Remediation pools cover short and long procedures instead of prioritizing only the shortest traces.

The stronger split audit groups `add(left,right)` with `add(operands)` and commuted addition/multiplication operands. The previous canonical-input-only audit missed these equivalences. Forty-seven validation and 47 test records overlap an earlier split and are now in `quarantined_heldout.jsonl`, outside training and scoring. All prior training records remain preserved. New examples inherit any existing group's split; otherwise their group gets a deterministic split. Validation panels count each equivalence group once per criterion. This check addresses these explicit equivalences, not arbitrary mathematical equivalence or all possible pretraining contamination.

## Run and dashboard

The pipeline and artifacts use `G:/ctfn-text/artifacts/v3_curriculum_v3`. The dashboard on port 8792 follows that pipeline's current specialist. Its displayed learning rate, round budget, attempt limit and example budget come from live policy metadata.

String remains queued, using its existing audited native dataset. It starts only after Maths passes and releases the GPU. No coordinator or unrelated tower is loaded for local Maths training.

## Windows file-lock recovery

A Windows access-denied error while replacing `status.json` interrupted the run during round 4. The saved checkpoint retained round 4, cursor 100, the optimizer and RNG states. Atomic writes now use unique temporary files and bounded retries for sharing conflicts, including checkpoint publication. The dashboard retries transient read conflicts. If the primary status file remains locked, status publication falls back to `status_fallback.json` without terminating training; the dashboard selects the newest timestamp and displays a warning. Disk-full and actual checkpoint failures still propagate rather than being silently ignored. A failed replacement preserves the previous complete checkpoint.
