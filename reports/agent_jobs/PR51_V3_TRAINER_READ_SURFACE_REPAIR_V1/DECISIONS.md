# Decisions

1. Do not expand the trainer allowlist from ten to twelve. Both extras are control
   metadata, not model input or trainer-safe metadata.
2. Place the ten declared payloads in `trainer/` and the signature plus declaration
   manifest in `control_plane/`; keep sealed and diagnostic domains separate.
3. Make `load_verified_trainer_inputs` the only read handoff. It returns bytes only
   after complete validation and does not return the packet path to the trainer.
4. Derive the declared set from an explicit role map rather than a hard-coded count.
5. Pin control bytes in the tracked descriptor; the control manifest pins the
   signature and trainer files. No generated file hashes itself.
6. Reject unexpected packet-root and trainer entries, dotfiles, all symlinked
   domains/files, hardlinks, directories, renames, missing files, duplicates,
   unsafe paths, type/role changes, lengths and hashes before any read handoff.
7. Preserve all V3 acquisition, semantic, ambiguity, diagnostic and outcome-market
   boundaries; require a new full independent acceptance rather than accepting this
   narrow repair as a substitute.
