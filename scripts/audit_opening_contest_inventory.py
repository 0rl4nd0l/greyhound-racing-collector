"""Audit already-authorized section structure and retained odds metadata, offline.

Reads access manifests before raw HTML; never parses final placing, odds,
predictions, protected bodies, databases, or prospective bundles. Section orders
are audited only for structure/coverage and never emitted or fitted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path


class SectionParser(HTMLParser):
    """Read only explicitly labelled in-running rows, without interpreting PIR."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.in_row = False
        self.in_title = False
        self.title = []
        self.boxes = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self.in_row = True
            self.title, self.boxes = [], []
        if self.in_row and tag == "td":
            self.in_title = "race__in-running__title" in attrs.get("class", "").split()
        if self.in_row and tag == "sprite-svg" and attrs.get("name", "").startswith("rug_"):
            self.boxes.append(attrs["name"][4:])

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)

    def handle_endtag(self, tag):
        if tag == "td":
            self.in_title = False
        if tag == "tr" and self.in_row:
            title = " ".join("".join(self.title).split())
            if title in {"1st Section", "2nd Section", "3rd Section"}:
                self.rows.append((title, tuple(self.boxes)))
            self.in_row = False


def section_structure(html):
    parser = SectionParser()
    parser.feed(html)
    groups = {}
    for label, boxes in parser.rows:
        groups.setdefault(label, []).append(boxes)
    first = groups.get("1st Section", [])
    valid = len(first) == 1 and 2 <= len(first[0]) <= 8 and len(set(first[0])) == len(first[0]) and all(b in "12345678" and len(b) == 1 for b in first[0])
    other = [boxes for label, boxes in parser.rows if label != "1st Section"]
    return {
        "labelled_section_rows": len(parser.rows),
        "first_section_present": bool(first),
        "first_section_unique_box_order": valid,
        "first_section_box_count": len(first[0]) if len(first) == 1 else None,
        "section_order_changes": bool(valid and any(set(x) == set(first[0]) and x != first[0] for x in other)),
        "duplicate_section_labels": any(len(v) != 1 for v in groups.values()),
    }


def read_json(path):
    return json.loads(Path(path).read_text())


def reference(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def checked_bytes(ref):
    body = Path(ref["path"]).read_bytes()
    if hashlib.sha256(body).hexdigest() != ref["sha256"]:
        raise ValueError("reference_hash_mismatch: " + ref["path"])
    return body


def run(args):
    # All access/split metadata is loaded and reconciled before any raw body.
    plan = read_json(args.plan)
    results = read_json(args.results)
    population = read_json(args.population)
    timing = read_json(args.timing)
    protected = set(plan["excluded_source_race_keys"]) | set(plan["test_scope"]["source_race_keys"])
    allowed = set(plan["allowed_source_race_keys"]) & set(results["allowed_source_race_keys"])
    membership = {row["source_race_key"]: row for row in population["roster_by_race"]}
    if len(membership) != 2351 or set(membership) - allowed or set(membership) & protected:
        raise ValueError("population_access_mismatch")
    raw_refs = {r["source_race_key"]: r for r in results["results"] if r["source_race_key"] in membership}
    if set(raw_refs) != set(membership):
        raise ValueError("result_membership_mismatch")
    mapping_path = Path(args.results).parent / "runner_mapping.jsonl"
    join_hashes = read_json(Path(args.results).parent / "artifacts.sha256.json")
    mapping_body = checked_bytes({"path": str(mapping_path), "sha256": join_hashes["runner_mapping.jsonl"]})
    mappings = defaultdict(dict)
    for line in mapping_body.decode().splitlines():
        mapping = json.loads(line)
        key = mapping["source_race_key"]
        if key not in membership or key in protected:
            raise ValueError("mapping_outside_admitted_scope")
        if mapping["disposition"] == "starter":
            box = str(mapping["final_box"])
            if box in mappings[key]:
                raise ValueError("duplicate_final_box")
            mappings[key][box] = mapping.get("source_native_dog_id")
    context_path = Path(args.context)
    context_hashes = read_json(context_path.parent / "artifacts.sha256.json")
    contexts = {}
    for line in checked_bytes({"path": str(context_path), "sha256": context_hashes[context_path.name]}).decode().splitlines():
        context = json.loads(line)
        key = context["source_race_key"]
        if key not in membership or key in protected:
            raise ValueError("context_outside_admitted_scope")
        value = context["static_target_context"].get("distance_m")
        if key in contexts and value != contexts[key]:
            raise ValueError("inconsistent_target_distance")
        contexts[key] = value
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows, by_track, event_support = [], {}, []
    for key in sorted(membership):
        source = raw_refs[key]
        html = checked_bytes(source["raw_html"]).decode("utf-8")
        structure = section_structure(html)
        parser = SectionParser()
        parser.feed(html)
        first_boxes = next((boxes for label, boxes in parser.rows if label == "1st Section"), ())
        exact_native = (structure["first_section_unique_box_order"] and set(first_boxes) == set(mappings[key])
                        and all(mappings[key].values()) and len(set(mappings[key].values())) == len(mappings[key]))
        structure["first_section_exact_native_field"] = bool(exact_native)
        if exact_native:
            event_support.append((key.rsplit(" - ",1)[1], (key.split(" - ")[1],contexts.get(key)), set(mappings[key].values())))
        row = {"source_race_key": key, "partition": membership[key]["partition"], "raw_html": source["raw_html"], "observed_at": source["observed_at"], **structure}
        # A matching box count is only a structural upper bound, never identity proof.
        row["first_section_count_matches_active_field"] = structure["first_section_unique_box_order"] and structure["first_section_box_count"] == membership[key]["active_starters"]
        rows.append(row)
        track = key.split(" - ")[1]
        by_track.setdefault(track, Counter())["races"] += 1
        by_track[track]["first_section_present"] += structure["first_section_present"]
        by_track[track]["count_matches_active_field"] += row["first_section_count_matches_active_field"]
    prior_support = {"runner_targets": 0, "with_prior_same_track_distance_call": 0, "whole_fields_with_prior_same_track_distance_call": 0,
                     "whole_fields_with_three_prior_same_track_distance_calls": 0}
    support_by_date = defaultdict(Counter)
    support_rows = []
    history = Counter()
    target_by_date = defaultdict(list)
    for key in membership:
        target_by_date[key.rsplit(" - ",1)[1]].append(key)
    events_by_date = defaultdict(list)
    for day, track, dogs in event_support:
        events_by_date[day].append((track,dogs))
    for day in sorted(target_by_date):
        for key in target_by_date[day]:
            track = (key.split(" - ")[1], contexts.get(key))
            support = [history[(track,dog)] if dog and track[1] and track[0] not in {"QOT", "RICH", "MURR"} else 0 for dog in mappings[key].values()]
            prior_support["runner_targets"] += len(support)
            prior_support["with_prior_same_track_distance_call"] += sum(n>0 for n in support)
            stable_roster = membership[key]["stable_primary_field"] and not membership[key]["promoted_reserves"]
            complete = bool(stable_roster and support and all(n>0 for n in support))
            complete3 = bool(stable_roster and support and all(n>=3 for n in support))
            prior_support["whole_fields_with_prior_same_track_distance_call"] += complete
            prior_support["whole_fields_with_three_prior_same_track_distance_calls"] += complete3
            support_rows.append({"source_race_key": key, "partition": membership[key]["partition"],
                                 "date": day, "track": track[0], "distance_m": track[1],
                                 "stable_primary_field_no_promotion": bool(stable_roster),
                                 "runners": len(support), "runners_with_prior_call": sum(n>0 for n in support),
                                 "minimum_prior_call_count": min(support) if support else 0,
                                 "complete_one_prior_call": complete, "complete_three_prior_calls": complete3})
            support_by_date[day]["races"] += 1
            support_by_date[day]["complete_prior_same_track_distance_call"] += complete
        # Simultaneous date update: never borrow another result from target day.
        for track,dogs in events_by_date[day]:
            for dog in dogs:
                history[(track,dog)] += 1
    # Revalidate retained metadata references by bytes only. No prospective
    # probabilities, runner histories, or labels are deserialized.
    captures = population["retained_capture_metadata"]
    metadata_refs = []
    for capture in captures:
        for kind in ("manifest", "completion", "sidecar"):
            checked_bytes(capture[kind])
            metadata_refs.append(capture[kind])
    identities = Counter(c["race_id"] for c in captures)
    for receipt in timing["timing"]["page_receipts"]:
        if receipt["source_race_key"] not in allowed or receipt["source_race_key"] in protected:
            raise ValueError("timing_access_mismatch")
        checked_bytes(receipt["raw_html"])
    result = {
        "schema_version": "opening_contest_inventory_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [reference(p) for p in (args.plan,args.results,args.population,args.timing)],
        "source_script": reference(__file__),
        "mapping_reference": reference(mapping_path),
        "context_reference": reference(context_path),
        "prior_date_same_track_distance_support_upper_bound": prior_support,
        "prior_date_same_track_distance_support_by_date": support_by_date,
        "raw_result_html_count": len(rows),
        "withheld_test_keys_checked": len(plan["test_scope"]["source_race_keys"]),
        "protected_raw_bodies_opened": 0,
        "section_counts": {k: sum(bool(r[k]) for r in rows) for k in ("first_section_present","first_section_unique_box_order","first_section_count_matches_active_field","section_order_changes","duplicate_section_labels","first_section_exact_native_field")},
        "section_counts_by_track": by_track,
        "timing_main_page_hashes_revalidated": len(timing["timing"]["page_receipts"]),
        "timing_semantic_status": timing["state"],
        "capture_metadata_inventory": {
            "snapshot_scope": "Existing fair-comparison population audit; October10 only; no new live inventory or payload admission",
            "captures": len(captures), "unique_races": len(identities),
            "races_with_multiple_captures": sum(n>1 for n in identities.values()),
            "references_hash_verified": len(metadata_refs),
            "metadata_sources": metadata_refs,
            "minimum_source_observed_lead_seconds": min(c["source_observed_lead_seconds"] for c in captures),
            "maximum_source_observed_lead_seconds": max(c["source_observed_lead_seconds"] for c in captures),
        },
        "fits": 0, "provider_requests": 0,
        "interpretation": "Section box order is newly inventoried measurement structure, not qualified early speed or historical decision-time availability. Single captures cannot support movement pairs. No model eligibility inferred.",
    }
    (output/"target_support.jsonl").write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in support_rows))
    (output/"section_inventory.jsonl").write_text("".join(json.dumps(row,sort_keys=True)+"\n" for row in rows))
    (output/"audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:result[k] for k in ("raw_result_html_count","section_counts","timing_main_page_hashes_revalidated","fits","provider_requests")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan","results","population","timing","context","output"):
        parser.add_argument("--"+name, required=True)
    run(parser.parse_args())
