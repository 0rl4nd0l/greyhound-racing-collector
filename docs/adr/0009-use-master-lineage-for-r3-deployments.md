# Use the master lineage for R3 deployments

Operator UI R3 deployments will be generated from the locally verified `origin/master` lineage, which contains the installed `e4f36999` snapshot plus the later manual-prediction safety work through GHU-063. The installed snapshot remains deployment evidence only, and the divergent legacy-Flask feature branch is not an R3 deployment source; reconstructing R3 from either would omit accepted safety contracts and create an obsolete source of truth.

Generation uses a clean, isolated checkout at the exact accepted commit and tree. The current feature worktree is preserved and cannot act as deployment source because it is both divergent and contains unrelated work.

Candidate `7b614325`, built directly from `e4f36999` with only legacy Flask endpoint fixes, is rejected under this decision and retained solely as evidence.

Legacy Flask hardening commits `debba152` and `f0b1c27f` remain outside replacement R3 candidates because they modify a different prediction interface and do not advance the R3 report-only job outcome.
