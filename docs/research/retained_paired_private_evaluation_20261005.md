# Fixed retrospective production-strength evaluation

This default-off adapter compares the independently qualified E02 production-parent probabilities at strengths 1.0 and 0.5. It neither derives new scores nor uses the different existing residual_half model as the half arm. Root approved the exact descriptive protocol in `research-continuation-20261005/paired-private-evaluation-interface-20261005.md` (SHA256 `72da2ea9fcfbf4f548401bb04cf08bdd1e2eb3465b6c9a246dc8634e2b571b8c`); delegated checkpoint SHA256 `5b9f9a984aface00a2793fac6af31bcb84e2cb62cbf0bffa35cde2f0ec480190` preserves the approval. Existing user approval covers private evaluation; root issues its exact new technical execution receipt after independent review. This file and the input scope are not executable authority.

The original 82-member October 1–3 population stays fixed. Expected categories are 73 complete-order win labels, one known-nonfinisher win label, and eight quarantines. Both arms use the same 74 eligible labels. A changed count fails; it does not silently choose a new subset. Original baseline claims, forecasts, sources and labels remain unchanged.

The new module adds three seams: strict `align_pair` authenticates common evidence and permutes sorted native-ID probabilities into the canonical original field order; `summarize_pairs` computes only race-weighted win log loss and multiclass Brier sum plus half-minus-full paired differences; `run_paired_evaluation` binds the exact stage inputs, claim and read limits. Negative differences mean lower descriptive loss. No clipping, numerical tolerance on the full-arm equality, significance testing, subgroup exploration, fitting or promotion occurs. The half probabilities were derived retrospectively, not sealed pre-jump; no new scientific or prospective membership is asserted.

The unchanged `retained_baseline_evaluation` module still verifies historical membership, all original input/bundle hashes, four sealed original forecasts, exact identity and labels. Its `win_target` handles verified tied winners by fractional mass and recognized nonfinish status without invented places. Its immutable SQLite reader has query-only mode, no WAL/journal acceptance, hash checks before/after, one-second query timeout, bounded rows and cell sizes. This adapter never invokes the old baseline claim/evaluation function or decodes its private metrics. Label-provenance validation rechecks the original separately verified successor mapping.

All input reads routed through the baseline's observer reader are wrapped in a process-local allowlist, size check and cumulative operation/byte budget, with restoration on failure. SHA pins come from sealed references/manifests; additional known pins are checked directly by the guard. The input scope retains the prior corrected full journal admission allowlist, including later-date admissions that authenticate the journal but do not enter the 82-member population. SQLite internal page IO and interpreter imports are not counted as artifact bytes; at most 73 immutable database reads and one known-nonfinish record, plus a whole-invocation deadline, bound this separate work. No transport is imported for invocation and the kernel network namespace denies access.

The scope/resource calculation preserves the previous baseline's finite 13,700-read / 951,058,432-byte ceiling and adds two exact passes over extra pair/control/input/production/manifest reads. Root must refresh only the new source file byte/hash pins after the final code freeze in an append-only scope successor. Source identity, exact clean Git commit, eleven implementation hashes and the exact Python executable hash are required by the new authority. Scope, protocol, expected categories and fixed claim path are immutable. Changing only output/authority cannot reset the exclusive claim.

Root-only execution contract:

```
python -B -m scripts.evaluate_retained_pairs
# DEFAULT_OFF: no authority or input paths opened
python -B -m scripts.evaluate_retained_pairs --execute \
  --authority /ABSOLUTE/ROOT/ISSUED/authority.json \
  --authority-sha256 EXACT_SHA256
```

Launch from the frozen checkout with `PYTHONPATH` and `PYTHONHOME` unset, `bwrap --unshare-net --ro-bind / /`, a writable empty private output parent and the one claim-file parent only, and a hard wall watchdog with cleanup margin. No service, provider, result-request, original data or live model changes. Private outputs are mode0600 in mode0700 directory. The terminal public receipt contains counts/categories/hashes only. Failure preserves the consumed claim and emits no completed metrics status; any partial private file remains failed evidence. Root separately qualifies the completed run before reading or releasing any metrics.

Fabricated checks exercise native-ID permutations, exact full-arm changes, parent/common/input/strength tampering, invalid probabilities, ties/nonfinish, guard limits/expiry/path hashes, native immutable SQLite composition, quarantine avoiding label access, all82/74/8 execution, claim reissue refusal and truthful failure output. These tests establish implementation behavior, not empirical advantage.

## Exact producer digest correction

The first root evaluation consumed its claim and stopped before labels at the first pair. Its immutable evidence remains under `paired-evaluation-execution-01`. The native pair producer's `controlled_retained_inputs.canonical` appends one newline before hashing; the baseline observer's canonical JSON does not. The initial adapter mistakenly used baseline bytes for the pair's self/common/roster digests. The successor uses exactly native newline bytes for those three pair checks, retaining unchanged baseline serialization and exact full-arm probability equality. It does not accept both formats. Fabricated fixtures now construct hashes independently from the producer contract, with a dedicated no-newline rejection case. Root must bind a separate corrected execution authority/scope/claim and preserve the failed predecessor.
