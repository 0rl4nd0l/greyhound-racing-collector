"""The two #193 recipes, with raw-card semantics and no training at inference."""
from __future__ import annotations

from datetime import date
import math

from scripts import build_form_only_v1_packet as canonical
from scripts.offline_form_packet import ALIASES, FEATURES, _metres, _number

SCHEMA = "offline_193_frozen_candidate_v1"
RECIPES = {
    "residual_box": {"features": [*FEATURES, "box_number"], "strength": 1.0},
    "residual_half": {"features": list(FEATURES), "strength": 0.5},
}
FEATURE_CONTRACT = {
    "history": "canonical accepted_history; date strictly before target; cap20",
    "source": "same retained normalized pre-race card as production; no DB augmentation",
    "rounding": "canonical feature_row decimal8 except recent_finish_best_5 and box",
    "distance": "explicit integer metres, optional m suffix; no history inference",
    "grade": "canonical category; no invented ordering",
    "venue": "legacy canonical aliases, including known pooled layouts",
    "recent_finish_best_5": "minimum known finish among first five accepted starts",
    "missingness": "null -> fitted median plus missing indicator",
    "box_number": "literal verified native box, not field-relative position",
}


def projected_dates(raw: bytes):
    """Project only DATE bytes, without decoding the other CSV field values.

    Strict CSV framing is checked, including quoted delimiters/newlines. Header
    has no outcome values. Protected history dates can therefore reject a card
    before the canonical parser materializes any historical result fields.
    """
    header, _, body = raw.partition(b"\n")
    delimiter = b"|" if header.count(b"|") > header.count(b",") else b","
    import csv
    names = next(csv.reader([header.decode("utf-8-sig")], delimiter=delimiter.decode()))
    indices = [i for i, name in enumerate(names) if name.strip() == "DATE"]
    if not indices:
        # A roster-only card is valid missing history, not fabricated starts.
        return []
    if len(indices) != 1:
        raise ValueError("ambiguous_history_date_column")
    target = indices[0]; field = 0; token = bytearray(); quoted = False; at_start = True; closed = False
    values = []; i = 0
    while i < len(body):
        byte = body[i:i+1]
        if quoted:
            if byte == b'"':
                if body[i+1:i+2] == b'"':
                    if field == target: token.extend(b'"')
                    i += 1
                else: quoted = False; closed = True
            elif field == target: token.extend(byte)
        elif byte == b'"' and at_start:
            quoted = True
        elif byte == delimiter or byte == b"\n":
            if field == target: values.append(bytes(token).strip().decode("ascii")); token.clear()
            field = 0 if byte == b"\n" else field + 1
            at_start = True; closed = False; i += 1; continue
        elif byte == b"\r":
            pass
        elif closed or byte == b'"':
            raise ValueError("unsupported_history_csv_framing")
        elif field == target:
            token.extend(byte)
        at_start = False; i += 1
    if quoted: raise ValueError("unterminated_history_csv_quote")
    if field == target and token: values.append(bytes(token).strip().decode("ascii"))
    return [date.fromisoformat(v) for v in values if v]


def card_features(raw, metadata, race_id, runners, *, captured_at, denied_history_intervals=()):
    target = date.fromisoformat(race_id.rsplit(" - ", 1)[1])
    for day in projected_dates(raw):
        if any(start <= day.isoformat() <= end for start, end in denied_history_intervals):
            raise ValueError("protected_history_date_before_outcome_decode")
        if day >= target or day > captured_at.date():
            raise ValueError("history_not_available_before_target")
    expected = sorted((r["box_number"], canonical.dog_token(r["display_name"])) for r in runners)
    if canonical.parse_card_target_roster_bytes(raw, source=race_id) != expected:
        raise ValueError("candidate_card_runner_mismatch")
    venue, _, grade, _ = canonical.target_metadata({"metadata": metadata}, race_id)
    distance = _metres(metadata.get("target_distance") or metadata.get("race_info", {}).get("distance"))
    blocks = canonical.parse_form_blocks_bytes(raw, source=race_id)
    output = []
    for box, token in expected:
        if token not in blocks: raise ValueError("candidate_history_block_missing")
        history, _ = canonical.accepted_history(blocks[token], target)
        values = canonical.feature_row(race_id, target, venue, distance, grade, len(expected), box, token, history)
        features = {name: _number(values.get(ALIASES.get(name, name))) for name in FEATURES}
        finishes = [h["finish"] for h in history[:5] if h["finish"] is not None]
        features.update(recent_finish_best_5=float(min(finishes)) if finishes else None, box_number=float(box))
        if any(v is not None and not math.isfinite(v) for v in features.values()):
            raise ValueError("candidate_nonfinite_feature")
        output.append({"box_number": box, "dog_token": token, "features": features})
    return output


def predict(model, features, market):
    """Scalar equivalent of #193's fitted median/scale/centred capped residual."""
    recipe = RECIPES[model["candidate_id"]]
    if model["schema_version"] != SCHEMA or model["recipe"] != {**recipe, "l2": 1.0, "cap": 0.35}:
        raise ValueError("candidate_recipe_changed")
    prep = model["fitted"]["prep"]; beta = model["fitted"]["beta"]
    names = recipe["features"]; width = len(names)*2
    if prep["names"] != names or prep["center"] is not True or len(beta) != width:
        raise ValueError("candidate_preprocessing_changed")
    if len(prep["median"]) != len(names) or len(prep["mean"]) != width or len(prep["scale"]) != width:
        raise ValueError("candidate_preprocessing_shape")
    if any(not math.isfinite(v) for v in beta + prep["median"] + prep["mean"] + prep["scale"]) or min(prep["scale"]) <= 0:
        raise ValueError("candidate_nonfinite_parameters")
    matrix = []
    for row in features:
        x = [row["features"][name] for name in names]
        if any(v is not None and (isinstance(v, bool) or not math.isfinite(v)) for v in x):
            raise ValueError("candidate_feature_invalid")
        expanded = [prep["median"][j] if v is None else v for j, v in enumerate(x)] + [float(v is None) for v in x]
        matrix.append([(v-prep["mean"][j])/prep["scale"][j] for j, v in enumerate(expanded)])
    if len(matrix) != len(market) or len(matrix) < 2 or any(not math.isfinite(p) or p <= 0 for p in market) or not math.isclose(math.fsum(market), 1, abs_tol=1e-12):
        raise ValueError("candidate_market_invalid")
    center = [math.fsum(row[j] for row in matrix)/len(matrix) for j in range(width)]
    logits = [math.log(p)+recipe["strength"]*.35*math.tanh(math.fsum((v-center[j])*beta[j] for j, v in enumerate(row))/.35) for row, p in zip(matrix, market)]
    exp = [math.exp(v-max(logits)) for v in logits]; total = math.fsum(exp)
    return [v/total for v in exp]
