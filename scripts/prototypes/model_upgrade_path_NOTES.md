# Prototype Notes

Question:
When the greyhound input data is finally clean enough, what should the next
model upgrade actually be: stronger tabular/ranking models, sequence neural
nets, or an LLM text sidecar?

Run:

`make prototype-model-upgrades`

What to look for:
- Does the state model hold tabular/ranking as the default core until the
  box-bias gate, clean labels, and non-box signal quality improve?
- Does sequence NN stay blocked until both label volume and ordered history
  depth are large enough?
- Does LLM stay blocked as a core predictor and only become a sidecar once
  provenance-safe text exists?

Verdict:
- TODO

Keep or delete:
- TODO after the prototype answers the question.
