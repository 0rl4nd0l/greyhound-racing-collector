# Validate R3 candidates before activation

An R3 candidate must pass source-lineage, artifact-identity, dependency, fixture-replay, and configuration validation before service activation. If a rejected candidate is activated by another actor, its running state does not reverse the rejection or qualify its predictions for acceptance. Replacement and activation remain separately controlled operations.

Generated R3 packages and source snapshots remain Unactivated R3 Deployment Candidates until their exact identities and contracts pass validation. Stale systemd failure state must not be attributed to a newly written candidate; `daemon-reload` and a single service restart occur only after candidate validation and require separate runtime authorization.
