# Use a forward-only per-race lifecycle

Each race advances through discovery, card collection, adaptive odds collection, evidence sealing, day closure, deferred prediction, result collection, and training-example eligibility. Prediction must commit or quarantine before result access begins; failures terminate in explicit quarantine states, and corrections create superseding records rather than moving completed state backward.
