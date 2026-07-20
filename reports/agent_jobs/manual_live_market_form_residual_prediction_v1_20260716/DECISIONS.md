# Decisions

1. Do not publish or use the form-only feature reconstruction.
2. Preserve the frozen model, candidate, strengths, and scorer unchanged.
3. Require exact precomputed, outcome-free system `shadow_feature_rows.json`
   plus its adjacent shadow and implementation manifests.
4. Keep production DB, network acquisition, runtime, persistence, service,
   deployment, promotion, betting, merge, and activation forbidden.
5. Require immutable single-read input hashing and complete APPENDED timestamp
   ordering in the revised adapter.
