# PR51 V3 trainer read-surface repair V1

Result: `DONE_WITH_RISK`.

This is one narrow descendant repair from exact rejected PR head
`91f854f5555bd1fd8ef411f2fadc108b41d2e5df`. The commit containing this report
is the new PR head. It closes the reproduced 12-readable-versus-10-declared
filesystem contract mismatch without widening the trainer allowlist.

The ten declared artifacts are isolated under `trainer/`. The two old extras,
`artifact-manifest.sha256` and `trainer_input_manifest.json`, are launcher
control metadata and now live under `control_plane/`. A trusted launcher verifies
the Git-tracked descriptor, exact four-domain packet root, exact control set,
exact trainer set, no-follow path types, lengths and hashes before returning only
the ten trainer payloads.

All acquisition counts and trainer bytes are unchanged. The repair opens no Jul
11-Aug 9 outcome, fits no model, constructs no market cohort, touches no runtime
or database, changes no PR 46-48 state, claims no edge, and does not activate,
merge or mark PR 51 ready.

Risk retained: the documented project environment cannot collect the broad
repository suite because `requirements.txt` does not provide `flask_compress`.
The 84 focused tests, compile, Ruff, 81% builder branch-inclusive coverage, two
clean deterministic real builds, exact-set probes, linkability scan, diagnostic
isolation and full diff review pass. The GitHub comprehensive checks remain the
publication gate, and a new full independent exact-head acceptance is mandatory.

Next independent-acceptance target:
`PR51_FORM_ONLY_V1_CONTRACT_REPAIR_V3_TRAINER_SURFACE_REPAIR_V1_INDEPENDENT_ACCEPTANCE_V1`
must review the published exact head from scratch and include every unfinished V3
check: complete trust-domain inventory; complete readable-set attacker scan;
descriptor/control/trainer root-chain verification; filesystem alias, traversal,
hardlink, symlink and discovery probes; immutable source and semantic-sidecar
binding; equal-precedence/equal-time ambiguity; diagnostic absence/move/corrupt/
mutation isolation; two clean rebuilds; domain hashes; parent/head broad-suite
adjudication; complete diff; PR-body truth; and draft/open/unmerged state.
