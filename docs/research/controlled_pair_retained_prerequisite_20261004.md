# Controlled-pair retained-input prerequisite (paused investigation)

No controlled derivation was executed. Implementation paused for the separately assigned controller repair.

The reviewed pure seam at feb3ec2a requires the exact original `market_form_residual_shadow_record_v3`, including `inputs.runners` and `inputs.provenance`, and exact replay equality. A native producer artifact is not that record.

Static evidence: `scripts/predict_market_form_residual.py:2798` creates the shadow record in memory; its output at 2812–2902 stores the original record key/checksum, source hashes, scoring-parity bindings and full/half output rows, but does not persist the original record inputs object. `scripts/predict_race_now.py:551` embeds this derived artifact in the sealed result. The first hash-verified live07 bundle manifest has retained form, sidecar, sealed history, feature rows and model files, but no standalone shadow-record file. This is a bounded first-bundle observation, not a statement that all 82 records lack it.

The existing 15-race metadata readiness report explicitly leaves opaque input hashing, numerical replay and full native identity reconstruction unverified. Baseline membership has 82 exact bundle references, but neither that membership nor historical result closure authenticates a missing shadow record.

Required next decision: either locate an original seal-bound native shadow record, or explicitly design an authenticated reconstruction whose checksum/key exactly match the pre-jump artifact. Such a reconstruction must be labeled reconstructed original computational content plus a new derivation timestamp, never a newly fabricated original forecast. It must authenticate native bundle/input/history/source/runtime/model pins, revalidate original timing and identity, preserve the entire development denominator, and reject missing/mismatched originals. No result or metric access is needed. The current pure module must not be fed a newly invented record merely because it can be rescored.

No payload values, outcomes or metrics were decoded, no new predictions created, and no provider/runtime changes made during this checkpoint.
