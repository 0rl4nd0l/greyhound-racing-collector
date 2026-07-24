"""Coherent ordered-finish probabilities from one Plackett--Luce distribution."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Mapping, Sequence


ORDERED_FINISH_CONTRACT = "plackett-luce-ordered-finish-v1"
MAX_EXACT_RUNNERS = 8


class OrderedFinishError(ValueError):
    """Latent strengths cannot define a trustworthy ordered-finish forecast."""


@dataclass(frozen=True, slots=True)
class OrderedFinishForecast:
    runner_ids: tuple[str, ...]
    order_probabilities: Mapping[tuple[str, ...], float]
    win: Mapping[str, float]
    top_2: Mapping[str, float]
    top_3: Mapping[str, float]
    exacta: Mapping[tuple[str, str], float]
    trifecta: Mapping[tuple[str, str, str], float]
    most_likely_orders: tuple[tuple[tuple[str, ...], float], ...]
    ranking: tuple[str, ...]

    def probability_of(self, order: Sequence[str]) -> float:
        return self.order_probabilities.get(tuple(order), 0.0)


def _sequential_probability(order: tuple[int, ...], weights: tuple[float, ...]) -> float:
    remaining = list(range(len(weights)))
    probability = 1.0
    for index in order[:-1]:
        denominator = math.fsum(weights[candidate] for candidate in remaining)
        probability *= weights[index] / denominator
        remaining.remove(index)
    return probability


def forecast_ordered_finish(
    runner_ids: Sequence[str], latent_strengths: Sequence[float], *, order_limit: int = 20
) -> OrderedFinishForecast:
    """Enumerate one stable PL distribution; intended for normal greyhound fields."""
    runners = tuple(runner_ids)
    strengths = tuple(float(value) for value in latent_strengths)
    if type(order_limit) is not int or order_limit <= 0:
        raise OrderedFinishError("order_limit must be a positive integer")
    if (
        not runners
        or len(runners) != len(strengths)
        or len(set(runners)) != len(runners)
        or any(type(runner) is not str or not runner for runner in runners)
        or any(not math.isfinite(value) for value in strengths)
    ):
        raise OrderedFinishError("runner identities and finite strengths must align")
    if len(runners) > MAX_EXACT_RUNNERS:
        raise OrderedFinishError(
            f"exact ordered-finish enumeration supports at most {MAX_EXACT_RUNNERS} runners"
        )
    maximum = max(strengths)
    weights = tuple(math.exp(max(value - maximum, -745.0)) for value in strengths)
    if not all(math.isfinite(value) and value > 0.0 for value in weights):
        raise OrderedFinishError("latent strengths are numerically unstable")

    indexed_orders = itertools.permutations(range(len(runners)))
    probabilities: dict[tuple[str, ...], float] = {}
    for indexed in indexed_orders:
        order = tuple(runners[index] for index in indexed)
        probability = _sequential_probability(indexed, weights)
        probabilities[order] = probability
    total = math.fsum(probabilities.values())
    if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise OrderedFinishError("ordered-finish distribution failed normalization")
    # Normalize once to remove enumeration rounding while preserving one distribution.
    probabilities = {order: value / total for order, value in probabilities.items()}
    # Every projection is recomputed from the final normalized distribution.
    win = {runner: 0.0 for runner in runners}
    top_2 = {runner: 0.0 for runner in runners}
    top_3 = {runner: 0.0 for runner in runners}
    exacta: dict[tuple[str, str], float] = {}
    trifecta: dict[tuple[str, str, str], float] = {}
    for order, probability in probabilities.items():
        win[order[0]] += probability
        for runner in order[:2]:
            top_2[runner] += probability
        for runner in order[:3]:
            top_3[runner] += probability
        if len(order) >= 2:
            exacta[order[:2]] = exacta.get(order[:2], 0.0) + probability
        if len(order) >= 3:
            trifecta[order[:3]] = trifecta.get(order[:3], 0.0) + probability
    likely = tuple(
        sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))[:order_limit]
    )
    ranking = tuple(sorted(runners, key=lambda runner: (-win[runner], runner)))
    return OrderedFinishForecast(
        runners, probabilities, win, top_2, top_3, exacta, trifecta, likely, ranking
    )


def ordered_finish_nll(forecast: OrderedFinishForecast, official_order: Sequence[str]) -> float:
    probability = forecast.probability_of(official_order)
    if probability <= 0.0:
        raise OrderedFinishError("official order is absent from the forecast distribution")
    return -math.log(probability)


def ordered_finish_from_probabilities(
    runner_ids: Sequence[str], order_probabilities: Mapping[tuple[str, ...], float]
) -> OrderedFinishForecast:
    """Rehydrate and validate every projection from one sealed full distribution."""
    runners = tuple(runner_ids)
    probabilities = dict(order_probabilities)
    expected = set(itertools.permutations(runners))
    if (
        not runners
        or len(set(runners)) != len(runners)
        or set(probabilities) != expected
        or any(not math.isfinite(value) or value < 0 for value in probabilities.values())
        or not math.isclose(math.fsum(probabilities.values()), 1.0, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise OrderedFinishError("sealed ordered distribution is incomplete or invalid")
    win = {runner: 0.0 for runner in runners}
    top_2 = {runner: 0.0 for runner in runners}
    top_3 = {runner: 0.0 for runner in runners}
    exacta: dict[tuple[str, str], float] = {}
    trifecta: dict[tuple[str, str, str], float] = {}
    for order, probability in probabilities.items():
        win[order[0]] += probability
        for runner in order[:2]:
            top_2[runner] += probability
        for runner in order[:3]:
            top_3[runner] += probability
        if len(order) >= 2:
            exacta[order[:2]] = exacta.get(order[:2], 0.0) + probability
        if len(order) >= 3:
            trifecta[order[:3]] = trifecta.get(order[:3], 0.0) + probability
    likely = tuple(sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))[:20])
    ranking = tuple(sorted(runners, key=lambda runner: (-win[runner], runner)))
    return OrderedFinishForecast(
        runners, probabilities, win, top_2, top_3, exacta, trifecta, likely, ranking
    )
