# V3.1 SIGReg probe

A three-round matched diagnostic started from the protected round-991 checkpoint. Both arms used the same saved weights, optimizer and RNG, the same 12,288 training examples in the same batch order, learning rate, clipping and replay. The only changed training setting was SIGReg coefficient: 0 versus 0.0003. Each endpoint was evaluated on the same 273 held-out stage questions.

Both arms achieved 269/273 correct answers and traces (98.53%), 214/216 comparison answers (99.07%), and 72/72 equality answers, including 12/12 original small-number equality questions. Last-round mean cross entropy was 0.03109469 without SIGReg and 0.03113440 with it. This short single-start experiment shows no measured accuracy disadvantage from SIGReg, but does not establish equivalence or long-term benefit. Updated data/range sampling plausibly explains the equality improvement; this experiment does not isolate those data changes.

Production weights were not overwritten by either trial. The original checkpoint was resumed with SIGReg 0.0003. Both endpoints, protected start, per-question results and schedule are saved under G:/ctfn-text/artifacts/v3_1/sigreg_probe_round_991. Use cftn_v3.sigreg_probe and scripts/summarize_sigreg_probe.py to inspect the procedure and results.
