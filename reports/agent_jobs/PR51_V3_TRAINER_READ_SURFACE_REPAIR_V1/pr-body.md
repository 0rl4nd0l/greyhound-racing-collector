## FORM_ONLY_V1 acquisition contract

This draft PR remains acquisition-only. Its current head is one normal direct
descendant of rejected head `91f854f5555bd1fd8ef411f2fadc108b41d2e5df`.

### Trainer read-surface repair

Independent acceptance reproduced 12 top-level regular files readable by the
trainer-side workflow while the manifest declared 10. The two extras were
`artifact-manifest.sha256` and `trainer_input_manifest.json`; both are launcher
control metadata, so this repair does not expand the trainer allowlist.

- `trainer/`: exactly 10 declared and actual files—five model-input datasets and
  five explicitly role-bound trainer-safe metadata files.
- `control_plane/`: exactly two launcher files, isolated from the trainer handoff.
- `sealed_validation/`: identity/source evidence, never trainer-readable.
- `non_authoritative_diagnostic/`: reconciliation only, never trainer input.

The trusted launcher validates the Git-tracked descriptor, exact four-domain
packet root, exact two-file control surface, exact ten-file trainer surface,
roles, regular single-link types, lengths, hashes, signature and aggregate before
returning only a filename-to-bytes mapping. Unexpected files (including a 13th),
dotfiles, symlinks, hardlinks, directories, renames, missing files, duplicate or
escaping declarations, type/role changes and byte/hash changes fail before any
trainer read handoff. Packet-root aliases and alternate discovery paths fail too.

The trust chain is non-self-referential: reviewed Git commit -> tracked descriptor
v3 -> two control hashes/aggregate -> trainer manifest -> trainer signature ->
ten trainer files.

### Preserved V3 evidence

- Authoritative counts: 1,267/8,914 candidate; 917/6,456 included; 67 sidecar-only
  exclusions; 88/617 outcome-unopened OOT.
- Domain aggregates: trainer `97967ab3...4e31`; control `1712d3d6...5462`;
  sealed `4ce9d105...26ed`; diagnostic `e5eaf492...995f`.
- Two clean builds are byte-identical; diagnostics do not change trainer bytes.
- Complete ten-file attacker scan has zero dog-token/digest, source-path, sealed
  alignment-key, cross-race identity or development-to-OOT identity intersections.
- Compile and Ruff pass; 84 focused tests pass; builder branch-inclusive coverage
  is 81%.
- The documented `requirements.txt` environment cannot collect the repository
  suite because `flask_compress` is missing. This is disclosed, not called green;
  the live GitHub comprehensive checks remain required publication evidence.

### Hard boundaries

No Jul 11-Aug 9 outcome was opened. No model was fit or evaluated, no market cohort
was created, no edge is claimed, no runtime/database/service was touched, and PRs
#46-#48 were not altered. This PR remains draft, open and unmerged; it is not an
activation or merge authorization.

A new full exact-head independent acceptance is mandatory. It must include every
unfinished V3 check: all trust domains and readable files, root-chain and attacker
view, traversal/link/default-discovery attacks, immutable and semantic source
binding, ambiguity gates, diagnostic absence/move/corrupt/mutation isolation, two
rebuilds and hashes, parent/head suite adjudication, complete diff/PR disclosure,
and draft/open/unmerged state.
