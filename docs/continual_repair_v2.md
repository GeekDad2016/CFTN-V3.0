# Canonical answer repair and continual learning

The legacy experiment reduced some token losses but produced zero exact answers on the fixed eight-question panels. Qwen targets often contained explanations or were truncated, while references used short answers.

The revised runner first repairs each active domain (Math, retrieval, Python, formal logic) using dataset reference targets. It then repeats Qwen-assisted learning until paused. Each domain block uses 1,024 distinct questions, 1,000 optimizer steps and 25% replay. A block is not 1,000 unique questions or a dataset epoch. Pools rotate over finite datasets; questions recur after those pools wrap.

Sources remain pinned GSM8K and BoolQ plus bounded synthetic Python and proof questions. The original held-out IDs are preserved. Synthetic training pools expand to 4,064 questions per domain; their coverage remains narrow. This is not general Python or PhD-level reasoning training.

Qwen receives explicit answer-format instructions. Only complete responses matching the reference under the bounded checker become teacher targets. Otherwise the canonical dataset answer is used and the rejected raw response and reason are recorded. Python is parsed and compared as an arithmetic polynomial; generated code is never executed. This is reference-checked distillation, not unrestricted learning from arbitrary teacher prose. No new teacher LoRA is introduced.

The student stays loaded between blocks. Every 200 steps, the runner measures 16 fixed held-out questions, four trained questions and eight retention questions, then atomically overwrites the unified checkpoint with weights, optimizer, RNG state and active-block progress. Restart resumes the last saved chunk. The previous repair checkpoint remains available. No experimental checkpoint is promoted to an accepted release.

Dispatcher/planner/synthesis updates wait for all four native domain panels to reach 75% with no measured retention drop. Dispatcher training then uses 1,000 steps and must reach 90% route accuracy before planner training. Planner validity must reach 90% before synthesis. These small panels are experiment gates, not broad capability certification.

The dashboard shows actual training targets and their source, held-out correctness, trained-question correctness, retention, question count, accepted teacher count and step budget. User tests remain queued and are serviced at checkpoint boundaries and while paused. The monitoring heartbeat remains paused independently of the training worker.
