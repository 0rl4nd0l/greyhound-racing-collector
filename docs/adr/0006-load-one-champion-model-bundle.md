# Load one champion model bundle

Production inference will resolve one durable champion pointer to one immutable, checksummed model bundle containing the model, forecast and feature contracts, derivation version, training configuration and cutoff, training-example identity, calibration, metrics, and runtime requirements. Environment overrides, registry ranking, filesystem recency, and mock-model fallbacks may not select production models; legacy artifacts require explicit conversion into the bundle contract.
