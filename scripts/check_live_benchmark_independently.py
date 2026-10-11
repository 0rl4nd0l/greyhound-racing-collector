#!/usr/bin/env python3
"""Independent retained-forecast checks; deliberately imports no benchmark scorer."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def compute_race(runners: list[dict[str, Any]]) -> dict[str, Any]:
    """Score a complete single-winner field without flooring or renormalizing model p."""
    if len(runners) < 2:
        raise ValueError("incomplete runner field")
    identities = [(str(row["runner_id"]), int(row["box"])) for row in runners]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate runner identity")
    if len({identity[0] for identity in identities}) != len(identities):
        raise ValueError("runner occupies multiple boxes")
    if len({identity[1] for identity in identities}) != len(identities):
        raise ValueError("duplicate box")
    p = [float(row["model_probability"]) for row in runners]
    odds = [float(row["decimal_odds"]) for row in runners]
    targets = [row["winner"] for row in runners]
    if any(isinstance(row["model_probability"], bool) for row in runners):
        raise ValueError("boolean model probability")
    if any(not math.isfinite(x) or x < 0 or x > 1 for x in p):
        raise ValueError("invalid model probability")
    if not math.isclose(math.fsum(p), 1, abs_tol=1e-6, rel_tol=0):
        raise ValueError("model probabilities do not sum to one")
    if any(not math.isfinite(x) or x <= 1 for x in odds):
        raise ValueError("invalid decimal odds")
    if any(type(y) not in (int, bool) or y not in (0, 1) for y in targets):
        raise ValueError("nonbinary outcome")
    if sum(targets) != 1:
        raise ValueError("expected one authorised winner")
    overround = math.fsum(1 / x for x in odds)
    q = [(1 / x) / overround for x in odds]
    winner = targets.index(1)

    def score(probs: list[float]) -> dict[str, float]:
        leaders = [i for i, value in enumerate(probs) if value == max(probs)]
        return {
            "log_loss": -math.log(probs[winner]) if probs[winner] else math.inf,
            "brier": math.fsum((value - targets[i]) ** 2 for i, value in enumerate(probs)),
            "top_accuracy": 1 / len(leaders) if winner in leaders else 0.0,
        }

    model, market = score(p), score(q)
    return {
        "model": model,
        "market": market,
        "difference": {name: model[name] - market[name] for name in model},
        "overround_factor": overround,
        "overround_excess": overround - 1,
        "market_probabilities": q,
        "runner_count": len(runners),
        "model_top_ids": sorted(identities[i][0] for i, x in enumerate(p) if x == max(p)),
        "market_top_ids": sorted(identities[i][0] for i, x in enumerate(q) if x == max(q)),
    }


def aggregate(races: list[dict[str, Any]]) -> dict[str, Any]:
    """Races carry race_id, date, model_version and checked runner records."""
    scores = [(race, compute_race(race["runners"])) for race in races]
    if len({race["race_id"] for race in races}) != len(races):
        raise ValueError("repeated primary race")

    def mean_score(selected: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
        n = len(selected)
        return {
            "races": n,
            "runners": sum(score["runner_count"] for _, score in selected),
            **{
                group: {
                    metric: (
                        math.fsum(score[group][metric] for _, score in selected) / n if n else None
                    )
                    for metric in ("log_loss", "brier", "top_accuracy")
                }
                for group in ("model", "market", "difference")
            },
        }

    dates = sorted({race["date"] for race in races})
    versions = sorted({race["model_version"] for race in races})
    bins: dict[str, dict[int, list[tuple[float, int]]]] = {
        "model": defaultdict(list),
        "market": defaultdict(list),
    }
    for race, score in scores:
        for index, row in enumerate(race["runners"]):
            for name, probability in (
                ("model", row["model_probability"]),
                ("market", score["market_probabilities"][index]),
            ):
                bins[name][min(9, int(probability * 10))].append((probability, row["winner"]))
    calibration = {}
    for name, cells in bins.items():
        total = sum(map(len, cells.values()))
        table = []
        for index in range(10):
            cell = cells[index]
            table.append(
                {
                    "bin": index,
                    "count": len(cell),
                    "mean_probability": math.fsum(p for p, _ in cell) / len(cell) if cell else None,
                    "observed_frequency": sum(y for _, y in cell) / len(cell) if cell else None,
                }
            )
        calibration[name] = {
            "bins": table,
            "runner_weighted_ece": (
                math.fsum(
                    row["count"] * abs(row["mean_probability"] - row["observed_frequency"])
                    for row in table
                    if row["count"]
                )
                / total
                if total
                else None
            ),
        }
    return {
        **mean_score(scores),
        "dates": dates,
        "model_versions": versions,
        "by_date": {
            day: mean_score([(race, score) for race, score in scores if race["date"] == day])
            for day in dates
        },
        "by_version": {
            version: mean_score(
                [(race, score) for race, score in scores if race["model_version"] == version]
            )
            for version in versions
        },
        "leave_one_date_out": {
            day: mean_score([(race, score) for race, score in scores if race["date"] != day])
            for day in dates
        },
        "calibration": calibration,
    }


def validate_record(record: dict[str, Any], diagnostic: bool = False) -> dict[str, Any]:
    """Validate integrated admission assertions and chronology independently."""
    from datetime import datetime

    required = {
        "allocation_status": "AUTHORISED_NONRESERVED",
        "forecast_type": "ORIGINAL_SEALED_LIVE",
        "field_status": "EXACT_UNCHANGED",
        "verified": True,
    }
    for key, value in required.items():
        if key == "field_status" and diagnostic and record.get(key) == "RESULT_FIELD_PARTIAL":
            continue
        if record.get(key) != value:
            raise ValueError(f"unacceptable {key}")
    for key in ("race_id", "date", "model_version", "prediction_id"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"missing identity: {key}")
    times = {}
    for key in ("quote_at", "cutoff_at", "predicted_at", "sealed_at", "jump_at", "result_at"):
        time = datetime.fromisoformat(record[key].replace("Z", "+00:00"))
        if time.tzinfo is None:
            raise ValueError("naive timestamp")
        times[key] = time
    if not (
        times["quote_at"]
        <= times["cutoff_at"]
        <= times["predicted_at"]
        <= times["sealed_at"]
        < times["jump_at"]
        <= times["result_at"]
    ):
        raise ValueError("invalid chronology")
    converted = dict(record)
    converted["runners"] = [
        {**row, "model_probability": row["probability"]} for row in record["runners"]
    ]
    compute_race(converted["runners"])
    return converted


def independent_date_intervals(races):
    """Bootstrap date-level sufficient statistics, without importing the root scorer."""
    import random

    dates = sorted({race["date"] for race in races})
    if len(dates) < 2:
        return None
    daily = []
    for date in dates:
        selected = [race for race in races if race["date"] == date]
        scores = [compute_race(race["runners"]) for race in selected]
        daily.append(
            (
                len(selected),
                {
                    metric: math.fsum(score["difference"][metric] for score in scores)
                    for metric in ("log_loss", "brier", "top_accuracy")
                },
            )
        )
    rng = random.Random(20261011)
    draws = {metric: [] for metric in ("log_loss", "brier", "top_accuracy")}
    for _ in range(10000):
        selected = [daily[rng.randrange(len(daily))] for _ in daily]
        count = sum(day[0] for day in selected)
        for metric in draws:
            draws[metric].append(math.fsum(day[1][metric] for day in selected) / count)
    return {metric: [sorted(values)[249], sorted(values)[9749]] for metric, values in draws.items()}


def compare_scorecard(
    records: list[dict[str, Any]], claimed: dict[str, Any], diagnostic: bool = False
) -> dict[str, Any]:
    """Reject mismatches between independently computed values and root scorecard."""
    checked = [validate_record(record, diagnostic=diagnostic) for record in records]
    versions = sorted({record["model_version"] for record in checked})
    if claimed["race_model_records"] != len(records):
        raise ValueError("race/model accounting mismatch")
    if claimed["unique_races"] != len({record["race_id"] for record in records}):
        raise ValueError("unique race accounting mismatch")
    if claimed["model_versions"] != versions:
        raise ValueError("model version accounting mismatch")
    expected_status = "SCORED" if records else "NO_ELIGIBLE_DECISION_TIME_COMPARISONS"
    if claimed["status"] != expected_status:
        raise ValueError("population status mismatch")

    def equivalent(actual: Any, expected: Any, path: str) -> None:
        if expected is None:
            if actual is not None:
                raise ValueError(f"expected null: {path}")
        elif isinstance(expected, (int, float)):
            if isinstance(actual, str) and actual in ("Infinity", "-Infinity"):
                actual = float(actual)
            if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"metric mismatch: {path}: {actual} != {expected}")
        elif actual != expected:
            raise ValueError(f"mismatch: {path}")

    def compare_summary(actual: dict[str, Any], expected: dict[str, Any], path: str) -> None:
        for key in ("races", "runners"):
            equivalent(actual[key], expected[key], f"{path}/{key}")
        for source, target in (("model", "model"), ("market", "market"), ("difference", "delta")):
            for metric, suffix in (
                ("log_loss", "log_loss"),
                ("brier", "brier"),
                ("top_accuracy", "top_credit"),
            ):
                equivalent(
                    actual[f"{target}_{suffix}"],
                    expected[source][metric],
                    f"{path}/{target}_{suffix}",
                )

    independent = {}
    for version in versions:
        subset = [record for record in checked if record["model_version"] == version]
        result = aggregate(subset)
        independent[version] = result
        claim = claimed["by_version"][version]
        compare_summary(claim, result, version)
        intervals = independent_date_intervals(subset)
        if intervals is None:
            if claim["uncertainty"]["intervals"] is not None:
                raise ValueError("uncertainty invented for fewer than two dates")
        else:
            for metric, bounds in intervals.items():
                suffix = "top_credit" if metric == "top_accuracy" else metric
                for i, bound in enumerate(bounds):
                    equivalent(
                        claim["uncertainty"]["intervals"]["delta_" + suffix][i],
                        bound,
                        version + "/date_interval/" + metric,
                    )
        independent[version]["independent_date_cluster_intervals"] = intervals
        for kind in ("by_date", "leave_one_date_out"):
            if set(claim[kind]) != set(result[kind]):
                raise ValueError(f"date accounting mismatch: {version}/{kind}")
            for date, metrics in result[kind].items():
                compare_summary(claim[kind][date], metrics, f"{version}/{kind}/{date}")
        for model in ("model", "market"):
            calibration = result["calibration"][model]
            claim_calibration = claim[f"{model}_calibration"]
            equivalent(
                claim_calibration["ece"],
                calibration["runner_weighted_ece"],
                f"{version}/{model}/ece",
            )
            for index, cell in enumerate(calibration["bins"]):
                for key in ("count", "mean_probability", "observed_frequency"):
                    equivalent(
                        claim_calibration["bins"][index][key],
                        cell[key],
                        f"{version}/{model}/bin{index}/{key}",
                    )
    expected_keys = {(r["race_id"], r["model_version"]) for r in records}
    actual_keys = [(r["race_id"], r["model_version"]) for r in claimed["per_race"]]
    if len(actual_keys) != len(expected_keys) or set(actual_keys) != expected_keys:
        raise ValueError("per-race accounting mismatch")
    for record in checked:
        result = compute_race(record["runners"])
        claim = next(
            row
            for row in claimed["per_race"]
            if (row["race_id"], row["model_version"])
            == (record["race_id"], record["model_version"])
        )
        for key in ("model", "market"):
            for metric in ("log_loss", "brier"):
                equivalent(
                    claim[f"{key}_{metric}"],
                    result[key][metric],
                    f"{record['race_id']}/{key}_{metric}",
                )
        equivalent(claim["overround"], result["overround_factor"], record["race_id"] + "/overround")
        for index, q in enumerate(result["market_probabilities"]):
            equivalent(claim["market_probabilities"][index], q, record["race_id"] + "/q")
    return {
        "status": "PASS",
        "race_model_records": len(records),
        "unique_races": len({record["race_id"] for record in records}),
        "by_version": independent,
        "limitation": "Source membership, exact original field and artifact provenance also require adapter review.",
    }


def check_membership(membership_path, boundary_path) -> dict[str, Any]:
    """Verify raw line hashes and allocation before decoding only authorised forecasts."""
    import hashlib
    import json
    from collections import Counter
    from pathlib import Path

    membership = json.loads(Path(membership_path).read_bytes())
    boundary = json.loads(Path(boundary_path).read_bytes())
    source_bytes = Path(membership["source"]).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != membership["source_sha256"]:
        raise ValueError("source ledger hash mismatch")
    lines = source_bytes.splitlines()
    if len(lines) != membership["total_records"] or len(lines) != len(membership["records"]):
        raise ValueError("ledger membership denominator mismatch")
    protected = set(boundary["canonical_protected_identity_keys"])
    counts = Counter()
    line_numbers = []
    for record in membership["records"]:
        line_numbers.append(record["line"])
        line = lines[record["line"] - 1]
        if hashlib.sha256(line).hexdigest() != record["line_sha256"]:
            raise ValueError("prediction line hash mismatch")
        key = record["canonical_race_key"]
        date = key.split("|")[0]
        forbidden = key in protected or any(
            date >= window["start"] and (window["end"] is None or date <= window["end"])
            for window in boundary["protected_windows"]
        )
        allowed = record["access_disposition"] == "AUTHORIZED_NONRESERVED_RETAINED_OPERATIONAL"
        if allowed == forbidden:
            raise ValueError("allocation decision mismatch")
        counts[record["access_disposition"]] += 1
        if allowed:
            original = json.loads(line)
            if (
                original["race_id"] != record["race_id"]
                or original["record_key"] != record["record_key"]
            ):
                raise ValueError("authorised prediction identity mismatch")
    if sorted(line_numbers) != list(range(1, len(lines) + 1)):
        raise ValueError("membership repeats or omits ledger line")
    for reference in boundary["references"]:
        if hashlib.sha256(Path(reference["path"]).read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError("protection reference hash mismatch")
    if not set(boundary["closed_114_original_keys"]) <= protected:
        raise ValueError("closed protected record omitted")
    return {
        "status": "PASS",
        "records": len(lines),
        "membership_counts": dict(counts),
        "source_hash_verified": membership["source_sha256"],
        "reserved_prediction_lines_decoded": 0,
        "protected_reference_hashes_verified": len(boundary["references"]),
        "canonical_reserved_keys": len(protected),
        "membership_sha256": hashlib.sha256(Path(membership_path).read_bytes()).hexdigest(),
        "access_boundary_sha256": hashlib.sha256(Path(boundary_path).read_bytes()).hexdigest(),
    }


def audit_integrated_sources(records, membership_path) -> dict[str, Any]:
    """Cross-check derived rows against exact admitted ledger and retained source bytes."""
    import hashlib
    import json
    import re
    from pathlib import Path

    membership = json.loads(Path(membership_path).read_bytes())
    lines = Path(membership["source"]).read_bytes().splitlines()
    allowed = {
        row["record_key"]: row
        for row in membership["records"]
        if row["access_disposition"] == "AUTHORIZED_NONRESERVED_RETAINED_OPERATIONAL"
    }
    references = set()
    for record in records:
        member = allowed.get(record["prediction_id"])
        if member is None:
            raise ValueError("integrated prediction is not admitted")
        raw = lines[member["line"] - 1]
        if hashlib.sha256(raw).hexdigest() != member["line_sha256"]:
            raise ValueError("integrated original line hash changed")
        original = json.loads(raw)
        if original["race_id"] != record["race_id"]:
            raise ValueError("integrated race identity mismatch")
        variant = record["model_version"].rsplit(":", 1)[-1]
        if variant not in ("full", "half") or not record["model_version"].startswith(
            original["model_sha256"] + ":"
        ):
            raise ValueError("integrated model identity mismatch")
        source_runners = {runner["runner_id"]: runner for runner in original["predictions"]}
        if set(source_runners) != {runner["runner_id"] for runner in record["runners"]}:
            raise ValueError("integrated runner field differs from original")
        for runner in record["runners"]:
            source = source_runners[runner["runner_id"]]
            if (
                runner["box"] != source["box_number"]
                or runner["probability"] != source[variant + "_probability"]
                or runner["decimal_odds"] != source["strict_win_odds"]
            ):
                raise ValueError("integrated prediction or odds changed")
        if (
            record["predicted_at"] != original["score_timestamp"]
            or record["jump_at"] != original["jump_timestamp"]
        ):
            raise ValueError("integrated original timing mismatch")
        for reference in record["artifacts"].values():
            if (
                hashlib.sha256(Path(reference["path"]).read_bytes()).hexdigest()
                != reference["sha256"]
            ):
                raise ValueError("integrated retained source hash changed")
            references.add((reference["path"], reference["sha256"]))
        evidence = record["result_evidence"]
        rows = evidence["runners"]
        by_box = {runner["box"]: runner for runner in record["runners"]}
        if len({row["box_number"] for row in rows}) != len(rows):
            raise ValueError("duplicate result box")
        winners = []
        for row in rows:
            runner = by_box.get(row["box_number"])
            if runner is None:
                raise ValueError("result has new runner")
            token = re.sub(r"[^A-Z0-9]", "", row["dog_name"].upper())
            expected_identity = f"{record['race_id']}|box:{row['box_number']}|dog:{token}"
            if (
                runner["runner_id"] != expected_identity
                or row["race_id"] != record["race_id"]
                or row["source"] != "thedogs_official"
            ):
                raise ValueError("result identity or source mismatch")
            if bool(row["is_winner"]) != bool(runner["winner"]):
                raise ValueError("integrated outcome differs from official result")
            if row["is_winner"]:
                if row["finish_position"] != 1:
                    raise ValueError("winner does not have first place")
                winners.append(row)
        if len(winners) != 1:
            raise ValueError("official single winner not confirmed")
        if record["field_status"] == "EXACT_UNCHANGED" and len(rows) != len(by_box):
            raise ValueError("partial result silently classified exact")
    return {
        "status": "PASS",
        "race_model_records": len(records),
        "unique_source_hashes_verified": len(references),
        "original_probabilities_and_odds_unchanged": True,
        "official_runner_identities_and_winner_checked": True,
    }


def check_selectors(records, plan, results, diagnostic=False) -> dict[str, Any]:
    """Reconstruct fixed temporal selectors independently, then audit their scorecards."""

    def value(record, selector):
        if selector == "model_confidence":
            return max(row["probability"] for row in record["runners"])
        if selector == "price_disagreement":
            return max(row["probability"] * row["decimal_odds"] - 1 for row in record["runners"])
        if selector == "relevant_history":
            return record.get("pre_race_quality", {}).get("minimum_starts_same_distance")
        raise ValueError("unknown selector")

    checks = 0
    for version in sorted({record["model_version"] for record in records}):
        subset = [record for record in records if record["model_version"] == version]
        dates = sorted({record["date"] for record in subset})
        frozen = plan["plans"][version]
        if len(dates) < 2:
            if frozen["status"] != "UNAVAILABLE_FEWER_THAN_TWO_DATES":
                raise ValueError("selector needs at least two dates")
            continue
        early = dates[: len(dates) // 2]
        late = dates[len(early) :]
        if frozen["development_dates"] != early or frozen["evaluation_dates"] != late:
            raise ValueError("selector temporal split mismatch")
        later = [record for record in subset if record["date"] in late]
        market_ranked = sorted(
            later,
            key=lambda record: (
                -max(
                    (1 / row["decimal_odds"])
                    / math.fsum(1 / r["decimal_odds"] for r in record["runners"])
                    for row in record["runners"]
                ),
                record["race_id"],
            ),
        )
        expected_rules = {
            (selector, coverage)
            for selector in ("model_confidence", "relevant_history", "price_disagreement")
            for coverage in (0.1, 0.25, 0.5, 1.0)
        }
        if {
            (rule["selector"], rule["target_coverage"]) for rule in frozen["rules"]
        } != expected_rules:
            raise ValueError("unexpected selector search or missing rule")
        for rule in frozen["rules"]:
            selector, coverage = rule["selector"], rule["target_coverage"]
            values = sorted(
                value(record, selector)
                for record in subset
                if record["date"] in early and value(record, selector) is not None
            )
            threshold = None
            if values and coverage != 1:
                rank = max(1, math.ceil(len(values) * (1 - coverage)))
                threshold = values[rank - 1]
            if rule["threshold"] != threshold:
                raise ValueError("selector threshold was not frozen from early predictors")
            selected = [
                record
                for record in later
                if coverage == 1
                or (
                    threshold is not None
                    and value(record, selector) is not None
                    and value(record, selector) >= threshold
                )
            ]
            claims = [
                claim
                for claim in results["comparisons"]
                if claim["model_version"] == version
                and claim["selector"] == selector
                and claim["target_coverage"] == coverage
            ]
            if len(claims) != 1:
                raise ValueError("selector comparison omitted or duplicated")
            claim = claims[0]
            if claim["selected_race_ids"] != [record["race_id"] for record in selected]:
                raise ValueError("selector chose different races")
            comparator = market_ranked[: len(selected)]
            if claim["market_confidence_race_ids"] != [record["race_id"] for record in comparator]:
                raise ValueError("market confidence comparator coverage or identity mismatch")
            if claim["selected_races"] != len(selected) or claim["evaluation_races"] != len(later):
                raise ValueError("selector coverage denominator mismatch")
            if claim["achieved_coverage"] != len(selected) / len(later):
                raise ValueError("selector achieved coverage mismatch")
            if results.get("status") == "FORECAST_SELECTION_ONLY_NO_OUTCOMES":
                if "selected" in claim or "market_confidence" in claim:
                    raise ValueError("outcome-free selector report contains scorecards")
            else:
                compare_scorecard(selected, claim["selected"], diagnostic=diagnostic)
                compare_scorecard(comparator, claim["market_confidence"], diagnostic=diagnostic)
            for record in later:
                dispositions = [
                    d
                    for d in results["dispositions"]
                    if d.get("model_version") == version
                    and d.get("race_id") == record["race_id"]
                    and d.get("selector") == selector
                    and d.get("target_coverage") == coverage
                ]
                if len(dispositions) != 1:
                    raise ValueError("selector disposition omitted or duplicated")
                v = value(record, selector)
                expected = (
                    "SELECTED"
                    if record in selected
                    else (
                        "PASS_MISSING_SEALED_QUALITY" if v is None or threshold is None else "PASS"
                    )
                )
                if dispositions[0]["disposition"] != expected:
                    raise ValueError("incorrect selector pass disposition")
            checks += 1
    if len(results["comparisons"]) != checks:
        raise ValueError("extra selector comparison")
    return {
        "status": "PASS",
        "comparisons_independently_checked": checks,
        "frozen_thresholds_membership_coverage_scores_and_pass_dispositions": True,
    }


def check_primary_selector_closure(records, selection, evaluation, diagnostic=False):
    """Keep the full-forecast split fixed when joining the available outcomes."""
    index = {(row["race_id"], row["model_version"]): row for row in records}
    if len(selection["comparisons"]) != len(evaluation["comparisons"]):
        raise ValueError("primary selector closure count mismatch")
    matched_counts = []
    for frozen, scored in zip(selection["comparisons"], evaluation["comparisons"]):
        for key, value in frozen.items():
            if scored.get(key) != value:
                raise ValueError("primary selector rule or cohort changed after outcome join")
        version = frozen["model_version"]
        selected = [
            index[(race_id, version)]
            for race_id in frozen["selected_race_ids"]
            if (race_id, version) in index
        ]
        market = [
            index[(race_id, version)]
            for race_id in frozen["market_confidence_race_ids"]
            if (race_id, version) in index
        ]
        if (
            scored["selected_outcomes"] != len(selected)
            or scored["selected_missing_outcomes"]
            != len(frozen["selected_race_ids"]) - len(selected)
            or scored["market_selector_outcomes"] != len(market)
            or scored["market_selector_missing_outcomes"]
            != len(frozen["market_confidence_race_ids"]) - len(market)
        ):
            raise ValueError("primary selector outcome coverage mismatch")
        compare_scorecard(selected, scored["selected"], diagnostic=diagnostic)
        compare_scorecard(market, scored["market_confidence"], diagnostic=diagnostic)
        matched_counts.append(
            {
                "selector": frozen["selector"],
                "target_coverage": frozen["target_coverage"],
                "selected_outcomes": len(selected),
                "market_outcomes": len(market),
            }
        )
    return {
        "status": "PASS",
        "comparisons_checked": len(matched_counts),
        "unchanged_full_forecast_selection": True,
        "outcome_counts": matched_counts,
    }


def main() -> None:
    import argparse
    import hashlib
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--scorecard", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--membership", type=Path)
    parser.add_argument("--boundary", type=Path)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--selector-plan", type=Path)
    parser.add_argument("--selectors", type=Path)
    parser.add_argument("--forecast-selection", type=Path)
    parser.add_argument("--primary-selector-evaluation", type=Path)
    args = parser.parse_args()
    membership_check = None
    if args.membership or args.boundary:
        if not args.membership or not args.boundary:
            raise ValueError("membership and boundary must be supplied together")
        membership_check = check_membership(args.membership, args.boundary)
    dataset_bytes = args.dataset.read_bytes()
    dataset = json.loads(dataset_bytes)
    scorecard = json.loads(args.scorecard.read_bytes())
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    protocol_hash = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    if dataset_hash != scorecard["dataset_sha256"]:
        raise ValueError("dataset hash mismatch")
    if dataset["protocol_sha256"] != protocol_hash or scorecard["protocol_sha256"] != protocol_hash:
        raise ValueError("protocol hash mismatch")
    expected_class = (
        "RESULT_FIELD_UNVERIFIED_DIAGNOSTIC" if args.diagnostic else "STRICT_DECISION_TIME"
    )
    if dataset.get("analysis_class", "STRICT_DECISION_TIME") != expected_class:
        raise ValueError("explicit diagnostic classification mismatch")
    if scorecard.get("analysis_class", "STRICT_DECISION_TIME") != expected_class:
        raise ValueError("scorecard classification mismatch")
    result = compare_scorecard(dataset["records"], scorecard, diagnostic=args.diagnostic)
    result["analysis_class"] = expected_class
    if args.selector_plan or args.selectors:
        if not args.selector_plan or not args.selectors:
            raise ValueError("selector plan and results must be supplied together")
        plan = json.loads(args.selector_plan.read_bytes())
        selectors = json.loads(args.selectors.read_bytes())
        if plan["dataset_sha256"] != dataset_hash or selectors["dataset_sha256"] != dataset_hash:
            raise ValueError("selector dataset binding mismatch")
        if selectors["plan_sha256"] != hashlib.sha256(args.selector_plan.read_bytes()).hexdigest():
            raise ValueError("selector plan hash mismatch")
        result["selector_check"] = check_selectors(
            dataset["records"], plan, selectors, args.diagnostic
        )
    if args.forecast_selection or args.primary_selector_evaluation:
        if not args.forecast_selection or not args.primary_selector_evaluation:
            raise ValueError("forecast selection and primary evaluation must be paired")
        selection = json.loads(args.forecast_selection.read_bytes())
        evaluation = json.loads(args.primary_selector_evaluation.read_bytes())
        if (
            evaluation["forecast_selection_sha256"]
            != hashlib.sha256(args.forecast_selection.read_bytes()).hexdigest()
        ):
            raise ValueError("primary forecast selection hash mismatch")
        result["primary_selector_check"] = check_primary_selector_closure(
            dataset["records"], selection, evaluation, args.diagnostic
        )
    result["membership_check"] = membership_check
    result["source_check"] = (
        audit_integrated_sources(dataset["records"], args.membership) if args.membership else None
    )
    result.update(
        dataset_sha256=dataset_hash,
        protocol_sha256=protocol_hash,
        scorecard_sha256=hashlib.sha256(args.scorecard.read_bytes()).hexdigest(),
    )

    def safe(value: Any) -> Any:
        if isinstance(value, float) and math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [safe(item) for item in value]
        return value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
