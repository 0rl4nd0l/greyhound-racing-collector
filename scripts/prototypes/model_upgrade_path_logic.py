"""PROTOTYPE ONLY.

Question: once the greyhound input data is trustworthy, when do stronger
tabular/ranking models, sequence neural nets, and LLM text sidecars become
justified? This prototype keeps the logic pure and in memory so the upgrade
decision can be pushed through edge cases before any real training work starts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


SIGNAL_QUALITIES = ("weak", "adequate", "strong")
BOX_BIAS_GATES = ("red", "yellow", "green")
ODDS_PROVENANCE = ("sparse", "partial", "strong")
HISTORY_DEPTHS = ("summary_only", "ordered_runs", "deep_sequences")
TEXT_CORPORA = ("none", "raw_notes", "structured_notes")
SOFT_DEBIAS_SIGNALS = ("none", "promising", "validated")

TABULAR_READY_RACES = 300
TABULAR_STRONG_RACES = 600
LLM_SIDECAR_READY_RACES = 1000
SEQUENCE_READY_RACES = 1500
SEQUENCE_READY_ROWS = 10000


@dataclass(frozen=True)
class UpgradeState:
    training_rows: int
    clean_labeled_races: int
    signal_quality: str
    box_bias_gate: str
    odds_provenance: str
    history_depth: str
    text_corpus: str
    soft_debias_signal: str


def initial_state(training_rows: int, clean_labeled_races: int) -> UpgradeState:
    return UpgradeState(
        training_rows=max(0, int(training_rows)),
        clean_labeled_races=max(0, int(clean_labeled_races)),
        signal_quality="weak",
        box_bias_gate="red",
        odds_provenance="sparse",
        history_depth="summary_only",
        text_corpus="none",
        soft_debias_signal="promising",
    )


def preset_state(name: str, seed_rows: int) -> UpgradeState:
    name = str(name).strip().lower()
    if name == "current":
        return initial_state(training_rows=seed_rows, clean_labeled_races=0)
    if name == "tabular":
        return UpgradeState(
            training_rows=max(seed_rows, 5000),
            clean_labeled_races=600,
            signal_quality="strong",
            box_bias_gate="green",
            odds_provenance="partial",
            history_depth="ordered_runs",
            text_corpus="none",
            soft_debias_signal="validated",
        )
    if name == "sequence":
        return UpgradeState(
            training_rows=max(seed_rows, 20000),
            clean_labeled_races=2000,
            signal_quality="strong",
            box_bias_gate="green",
            odds_provenance="strong",
            history_depth="deep_sequences",
            text_corpus="none",
            soft_debias_signal="validated",
        )
    if name == "llm":
        return UpgradeState(
            training_rows=max(seed_rows, 20000),
            clean_labeled_races=2000,
            signal_quality="strong",
            box_bias_gate="green",
            odds_provenance="strong",
            history_depth="deep_sequences",
            text_corpus="structured_notes",
            soft_debias_signal="validated",
        )
    raise ValueError(f"unknown preset: {name}")


def cycle_value(current: str, values: tuple[str, ...]) -> str:
    index = values.index(current)
    return values[(index + 1) % len(values)]


def reduce_state(state: UpgradeState, action: str, seed_rows: int) -> UpgradeState:
    action = str(action).strip().lower()

    if action == "1":
        return preset_state("current", seed_rows)
    if action == "2":
        return preset_state("tabular", seed_rows)
    if action == "3":
        return preset_state("sequence", seed_rows)
    if action == "4":
        return preset_state("llm", seed_rows)
    if action == "l":
        return replace(state, clean_labeled_races=state.clean_labeled_races + 100)
    if action == "k":
        return replace(state, clean_labeled_races=max(0, state.clean_labeled_races - 100))
    if action == "u":
        return replace(state, training_rows=state.training_rows + 1000)
    if action == "j":
        return replace(state, training_rows=max(0, state.training_rows - 1000))
    if action == "s":
        return replace(
            state, signal_quality=cycle_value(state.signal_quality, SIGNAL_QUALITIES)
        )
    if action == "b":
        return replace(state, box_bias_gate=cycle_value(state.box_bias_gate, BOX_BIAS_GATES))
    if action == "o":
        return replace(
            state, odds_provenance=cycle_value(state.odds_provenance, ODDS_PROVENANCE)
        )
    if action == "h":
        return replace(
            state, history_depth=cycle_value(state.history_depth, HISTORY_DEPTHS)
        )
    if action == "t":
        return replace(state, text_corpus=cycle_value(state.text_corpus, TEXT_CORPORA))
    if action == "d":
        return replace(
            state,
            soft_debias_signal=cycle_value(
                state.soft_debias_signal, SOFT_DEBIAS_SIGNALS
            ),
        )
    return state


def _tabular_blockers(state: UpgradeState) -> list[str]:
    blockers: list[str] = []
    if state.signal_quality == "weak":
        blockers.append("non-box signal is still weak/defaulted")
    if state.box_bias_gate == "red":
        blockers.append("box-bias gate is still red")
    if state.clean_labeled_races < TABULAR_READY_RACES:
        blockers.append(
            f"need at least {TABULAR_READY_RACES} clean labeled races, have {state.clean_labeled_races}"
        )
    if state.soft_debias_signal == "none":
        blockers.append("no promising debias signal to seed the next challenger")
    return blockers


def _sequence_blockers(state: UpgradeState, tabular_ready: bool) -> list[str]:
    blockers: list[str] = []
    if not tabular_ready:
        blockers.append("tabular/ranking challenger is not ready yet")
    if state.clean_labeled_races < SEQUENCE_READY_RACES:
        blockers.append(
            f"need at least {SEQUENCE_READY_RACES} clean labeled races, have {state.clean_labeled_races}"
        )
    if state.training_rows < SEQUENCE_READY_ROWS:
        blockers.append(
            f"need at least {SEQUENCE_READY_ROWS} clean training rows, have {state.training_rows}"
        )
    if state.history_depth != "deep_sequences":
        blockers.append("ordered per-dog run sequences are not deep enough")
    return blockers


def _llm_sidecar_blockers(state: UpgradeState, tabular_ready: bool) -> list[str]:
    blockers: list[str] = []
    if not tabular_ready:
        blockers.append("tabular/ranking core is not ready yet")
    if state.clean_labeled_races < LLM_SIDECAR_READY_RACES:
        blockers.append(
            f"need at least {LLM_SIDECAR_READY_RACES} clean labeled races, have {state.clean_labeled_races}"
        )
    if state.signal_quality != "strong":
        blockers.append("core feature packet is not strong enough yet")
    if state.text_corpus != "structured_notes":
        blockers.append("text corpus is missing or not structured into provenance-safe notes")
    return blockers


def evaluate_state(state: UpgradeState) -> dict:
    tabular_blockers = _tabular_blockers(state)
    tabular_ready = not tabular_blockers

    sequence_blockers = _sequence_blockers(state, tabular_ready)
    sequence_ready = not sequence_blockers

    llm_sidecar_blockers = _llm_sidecar_blockers(state, tabular_ready)
    llm_sidecar_ready = not llm_sidecar_blockers

    if not tabular_ready:
        recommendation = "Hold: fix feature packet and collect more clean official labels."
        next_core = "stay on calibrated tabular baseline until evidence is cleaner"
    elif sequence_ready and llm_sidecar_ready:
        recommendation = (
            "Core should remain tabular/ranking. Run bounded challengers for both "
            "sequence NN and LLM text sidecar."
        )
        next_core = "calibrated tabular/ranking with two bounded challenger lanes"
    elif sequence_ready:
        recommendation = (
            "Core should remain tabular/ranking. Sequence NN is now justified as a "
            "bounded challenger."
        )
        next_core = "tabular/ranking core plus sequence challenger"
    elif llm_sidecar_ready:
        recommendation = (
            "Core should remain tabular/ranking. LLM text sidecar is now justified "
            "as a bounded challenger."
        )
        next_core = "tabular/ranking core plus text-sidecar challenger"
    else:
        recommendation = (
            "Next upgrade is still stronger tabular/ranking challengers, not NN or LLM core."
        )
        next_core = "calibrated tabular/ranking challengers"

    ready_paths = []
    if tabular_ready:
        ready_paths.append("calibrated tabular/ranking challengers")
    if sequence_ready:
        ready_paths.append("sequence NN challenger")
    if llm_sidecar_ready:
        ready_paths.append("LLM text-sidecar challenger")

    return {
        "state": asdict(state),
        "recommendation": recommendation,
        "next_core": next_core,
        "ready_paths": ready_paths,
        "candidate_status": {
            "tabular_ranking": {
                "ready": tabular_ready,
                "blockers": tabular_blockers,
                "why": "best next step for structured race data once labels and signal are clean",
            },
            "sequence_nn": {
                "ready": sequence_ready,
                "blockers": sequence_blockers,
                "why": "only justified after strong tabular baseline and much deeper clean sequence data",
            },
            "llm_text_sidecar": {
                "ready": llm_sidecar_ready,
                "blockers": llm_sidecar_blockers,
                "why": "text helper for steward/form notes, not the primary predictor",
            },
            "llm_core_predictor": {
                "ready": False,
                "blockers": [
                    "core problem is structured tabular ranking, not free-form language generation"
                ],
                "why": "keep LLMs out of the core win-probability engine",
            },
        },
        "thresholds": {
            "tabular_ready_races": TABULAR_READY_RACES,
            "tabular_strong_races": TABULAR_STRONG_RACES,
            "llm_sidecar_ready_races": LLM_SIDECAR_READY_RACES,
            "sequence_ready_races": SEQUENCE_READY_RACES,
            "sequence_ready_rows": SEQUENCE_READY_ROWS,
        },
    }
