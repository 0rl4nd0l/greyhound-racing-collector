# Independent standards and implementation review

Reviewed `git diff 27fe8945...2637324c`, then independently rechecked integrated repairs through `6c1aca4d`, using the installed code-review skill. Standards sources: `AGENTS.md`, `CONTEXT.md`, `pyproject.toml`, and `file_naming_standards/NAMING_STANDARDS.md`. The current user request authorizes offline experiments. No provider or service actions were taken.

## Findings and disposition

- **Resolved hard contract, P2 — malformed race acceptance.** The original `scripts/run_retrospective_sp_benchmark.py` accepted additional active result runners, multiple winners and nonnormalized probabilities; `race_collection/research_correction_diagnostics.py` accepted labels `[1,1,-1]`. These contradicted the `CONTEXT.md` Evaluation-eligible Race requirement for an unambiguous outcome and its coherent-probability definition. Synthetic reproductions confirmed the defects. Integrated repairs require an exact active field and complete finishing order, binary unique winners and normalized finite probabilities; regression tests pass.
- **Resolved implementation integrity, P2 — incomplete input validation.** Original benchmark extension loading decoded base predictions without checking their manifest, and preparation decoded unverified runner mappings with silent duplicate-key overwrite. The immutable provenance contract (`CONTEXT.md`, Training Example and Prediction Provenance) requires dependable input identity. Integrated repairs check required manifest entries and hashes before decoding and reject duplicate mappings. No current score corruption was found.
- **Nonblocking heuristic — possible Duplicated Code.** Benchmark `run` and `add_dynamic` repeat chronological meta-fit, application, intent-recording and summary loops. A shared bounded runner could reduce future divergence in chronology or failure handling. This is maintainability advice, not a hard requirement.

## Verification and acceptance

**30 focused tests passed in 0.52 seconds** at `6c1aca4d`: the four new test modules, with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, single-thread BLAS, and `pytest --noconftest -o addopts=''`. Initial ordinary pytest imported unrelated application fixtures; its owned process was terminated before isolated validation.

After reading access metadata, independently checked all 19 dynamic, 53 base-market and 34 extension artifact hashes. Both market prediction files contain 1,631 distinct authorized development races and valid boxes, winners and probabilities. The benchmark repair audit rechecked 2,351 source bodies and reproduced 32 metrics with no changes or optimizer calls. No reserved labels were opened.

Replacement-mode review confirmed the six removed averages actually leave the fitted feature list, with unchanged state computation and downstream objective. Prior-date batching and chronology tests pass. **No unresolved correctness blockers; one nonblocking duplication heuristic.** Scientific validity and interpretation remain the separate methodology review's responsibility.
