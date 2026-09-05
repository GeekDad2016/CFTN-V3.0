# V3.0 provenance

V3.0 is an independent implementation informed by CFTN V2's typed dispatch,
independent wake gates, bidirectional message bridges, and causal ablations.
Inspected reference: `ctfn_text_build` commit
`abf1002a7c817a1648db4a615960d5b4cb8c5310`.

No mutable imports, checkpoint dependencies, copied datasets, or teacher weights
are required from V2 or V12. All twelve specialist parameter sets initialize
randomly. The pinned coordinator is the sole pretrained deployed component.

The bootstrap data generator is new and deliberately bounded. Its English and
Romanian templates are parallel specifications, not machine-translated evaluation
labels. It does not demonstrate unrestricted translation, science, commonsense,
long-context reasoning, code competence, or mathematical expertise.

Further data imports must carry a revision, source/license record, independent
verification, and semantic split identity. Existing V2 dataset license notes
must not be assumed to establish current permissions for V3.0.
