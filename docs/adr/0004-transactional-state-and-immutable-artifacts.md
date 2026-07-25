# Separate transactional workflow state from immutable artifacts

The Race Collection Service will keep lifecycle transitions, operation IDs, leases, batch barriers, quarantine states, champion pointers, and provenance indexes in one transactional state store. Raw and normalized evidence, predictions, results, manifests, scorecards, and model bundles will be immutable content-addressed artifacts referenced by checksum; JSON reports are human-readable projections and never authoritative workflow state.
