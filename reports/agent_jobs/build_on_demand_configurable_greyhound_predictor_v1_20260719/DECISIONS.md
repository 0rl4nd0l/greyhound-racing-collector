# Decisions

1. Keep output research-only and isolated. The command has no shadow append,
   prediction writer, betting output, production registry, daemon, or timer path.
2. Treat every existing collector lock as busy. The command uses `O_EXCL`, never
   evaluates staleness, and releases only an exact owner-and-inode match.
3. Seal history through SQLite `mode=ro` plus `query_only`, excluding the target,
   same-day, future, missing, duplicate, and relevant malformed-date identities.
4. Resolve finite model aliases before validating canonical JSON against an
   exact model schema. Frozen coefficients are copied and hash-recorded, not
   mutated.
5. Reuse scoring only. PR #47 `score_from_artifacts` is exact and tested; PR #46
   `append_shadow_record` is unused, unchanged, and explicitly blocked.
6. Do not run the live command without a separately named race and explicit
   owner authorization.
