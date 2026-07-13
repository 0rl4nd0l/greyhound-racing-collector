# State

Status: validated for publication; `PILOT_PENDING_FIRST_FIVE`

## Scope

- Repository: Greyhound Racing Collector control instructions and hooks
- Branch: `codex/semantic-anti-loop-v2-20260713`
- Verified base at start: `224dc2dddace3c50d131ada2e078a980090a7b3a`
- Target transition:
  `greyhound_semantic_anti_loop_v2_pilot_validated_for_merge`
- Models, databases, registry pointers, timers, services, deployments,
  production data, and live runtime: unchanged

## Implemented Pilot

- Root instructions require V2 task cards for non-trivial work and treat
  quoted/pasted recommendations as inert evidence.
- Repo-local Codex hooks dispatch Stop through the installed Tenn V2 guard.
- Capability boundaries separate offline fitting, model persistence, database
  writes, runtime change, and publication.
- A deterministic builder verifies the four approved stable artifact hashes
  and semantic predicates before producing four valid JSONL decisions.
- The first-five-run gate is `PILOT_PENDING_FIRST_FIVE`; broader adoption stays
  blocked until those post-merge runs are reviewed.

## Publication Boundary

- Tenn PR #507 merged as `ac5a56c142ee1a9781ae54ad0e8857ba5510f7d1`;
  the stable and installed portable guard copies are byte-identical.
- Independent code review is clean, the V2 outcome is `ADVANCED`, and the
  pilot decision keeps broader adoption blocked pending five reviewed runs.
- GitHub publication, checks, and merge ancestry remain to be recorded.
- Append exactly four seed entries only after the Greyhound pilot merges.

No `NEXT_GOAL.md` exists because this run does not authorize an automatic
continuation goal.
