"""
GPTService: Thin adapter over utils.openai_wrapper.OpenAIWrapper
- Centralizes GPT usage for the /api/gpt/* endpoints
- Avoids reliance on archived enhancers; keeps endpoint contracts intact
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.openai_config import OpenAIConfig, get_openai_config


@dataclass
class GPTServiceInit:
    available: bool
    model: str
    error: Optional[str] = None


class GPTService:
    def __init__(self) -> None:
        self.cfg: OpenAIConfig = get_openai_config()
        self.model: str = self.cfg.model
        self._error: Optional[str] = None
        self._available: bool = False
        self._wrapper = None
        self._client = None

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self._error = "OPENAI_API_KEY not configured"
            self._available = False
            return

        try:
            # OpenAI Python SDK v1.x style client
            from openai import OpenAI  # type: ignore

            # Prefer constructing with an explicit httpx client to avoid
            # version/env mismatches (e.g., proxies kw) in OpenAI internals
            try:
                import httpx  # type: ignore

                http_client = httpx.Client(timeout=30.0)
                self._client = OpenAI(http_client=http_client)
            except Exception:
                # Fallback: default construction
                self._client = OpenAI()

            from utils.openai_wrapper import OpenAIWrapper

            self._wrapper = OpenAIWrapper(self._client, self.cfg)
            self._available = True
        except Exception as e:
            # Last-resort retry: try plain client once if http_client path failed due to signature mismatch
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI()
                from utils.openai_wrapper import OpenAIWrapper

                self._wrapper = OpenAIWrapper(self._client, self.cfg)
                self._available = True
                self._error = None
            except Exception as e2:
                self._error = str(e2 if e2 else e)
                self._available = False

    @property
    def gpt_available(self) -> bool:
        return self._available and (self._wrapper is not None)

    @property
    def init(self) -> GPTServiceInit:
        return GPTServiceInit(
            available=self.gpt_available, model=self.model, error=self._error
        )

    # Internal helper to get JSON response with token usage
    def _respond_json_with_usage(
        self, prompt: str, system: Optional[str] = None
    ) -> Dict[str, Any]:
        if not self.gpt_available:
            return {"error": self._error or "OpenAI unavailable"}
        # Use respond_text so we capture usage; enforce JSON-only in system message
        try:
            from utils.openai_wrapper import OpenAIWrapper  # for type hints only

            sysmsg = (
                (system + "\nReturn valid JSON only.")
                if system
                else "Return valid JSON only."
            )
            resp = self._wrapper.respond_text(prompt=prompt, system=sysmsg)
            text = resp.text or "{}"
            data = json.loads(text)
            # Attach tokens if available
            usage = resp.usage or {}
            try:
                data.setdefault("_meta", {})["tokens_used"] = usage.get("total_tokens")
            except Exception:
                pass
            return data
        except Exception as e:
            return {"error": f"OpenAI error: {e}"}

    def enhance_predictions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Rerank existing ML predictions using GPT with a strict JSON contract.

        Input payload schema (minimal):
          {
            "race_info": { ... optional ... },
            "runners": [
              {"dog_name": str, "box_number": int|None, "win_prob": float, ...}, ...
            ]
          }

        Returns:
          {"scores": [{"dog_name": str, "gpt_score": float}, ...], "_meta": {"tokens_used": int}?}
        """
        try:
            if not isinstance(payload, dict):
                return {"scores": []}
            runners = payload.get("runners") or []
            if not isinstance(runners, list) or len(runners) == 0:
                return {"scores": []}
        except Exception:
            return {"scores": []}

        if not self.gpt_available:
            # Degrade: return empty scores so caller can skip blending
            return {"scores": []}

        # Build a compact prompt for strict JSON output
        try:
            names = [str((r.get("dog_name") or "")).strip() for r in runners]
            base = []
            for r in runners:
                try:
                    v = r.get("win_prob")
                    if v is None:
                        v = (
                            r.get("final_score")
                            or r.get("win_probability")
                            or r.get("confidence")
                            or 0.0
                        )
                    v = float(v)
                except Exception:
                    v = 0.0
                # Convert percentages (>1) to 0-1
                if v > 1.5:
                    v = v / 100.0
                if v < 0:
                    v = 0.0
                base.append(v)
            # Normalize base to sum to 1 for reference
            s = sum(base)
            if s > 0:
                base_norm = [x / s for x in base]
            else:
                base_norm = [1.0 / len(base)] * len(base)

            system = (
                "You are a racing reranker.\n"
                "Return strict JSON only. Do not include prose.\n"
                "Given runner names and baseline probabilities, produce a JSON object with key 'scores'\n"
                "containing an array of objects: {dog_name, gpt_score}. The gpt_score must be in [0,1] and all gpt_scores must sum to 1.\n"
                "Make only light adjustments to baseline; keep ordering similar unless strong cues exist."
            )
            # Provide minimal context; keep token usage small
            runners_lines = [
                f"{i+1}. {names[i]} (base={base_norm[i]:.3f})"
                for i in range(len(names))
            ]
            prompt = (
                "Runners (name with baseline):\n"
                + "\n".join(runners_lines)
                + '\n\nReturn JSON with key \'scores\' only. Example: {"scores":[{"dog_name":"NAME","gpt_score":0.123}]}'
            )
            data = self._respond_json_with_usage(prompt=prompt, system=system)
            # Validate shape and coerce
            if not isinstance(data, dict):
                return {"scores": []}
            scores = data.get("scores")
            if not isinstance(scores, list) or len(scores) == 0:
                return {"scores": []}
            out = []
            for obj in scores:
                try:
                    nm = str(obj.get("dog_name") or "").strip()
                    sc = float(obj.get("gpt_score") or 0.0)
                except Exception:
                    continue
                if not nm:
                    continue
                if sc < 0:
                    sc = 0.0
                if sc > 1:
                    sc = sc / 100.0 if sc > 1.5 else 1.0
                out.append({"dog_name": nm, "gpt_score": sc})
            if not out:
                return {"scores": []}
            tot = sum(o["gpt_score"] for o in out)
            if tot <= 0:
                # assign equal
                eq = 1.0 / len(out)
                for o in out:
                    o["gpt_score"] = eq
            else:
                for o in out:
                    o["gpt_score"] = o["gpt_score"] / tot
            # carry forward tokens if present
            meta_tokens = None
            try:
                meta_tokens = (data.get("_meta") or {}).get("tokens_used")
            except Exception:
                meta_tokens = None
            resp = {"scores": out}
            if meta_tokens is not None:
                resp["_meta"] = {"tokens_used": meta_tokens}
            return resp
        except Exception:
            return {"scores": []}

    def _collect_csv_context(self, race_file_path: str) -> Dict[str, Any]:
        """Extract lightweight context from the CSV and filename metadata.
        Returns keys: participants (list[str]), distance, grade, venue, race_date, race_number.
        All fields are optional; returns best-effort values.
        """
        ctx: Dict[str, Any] = {
            "participants": [],
            "distance": None,
            "grade": None,
            "venue": None,
            "race_date": None,
            "race_number": None,
        }
        try:
            from utils.csv_metadata import parse_race_csv_meta  # type: ignore

            meta = parse_race_csv_meta(race_file_path)
            if isinstance(meta, dict) and meta.get("status") == "success":
                ctx["venue"] = meta.get("venue") or None
                ctx["race_date"] = meta.get("race_date") or None
                ctx["race_number"] = meta.get("race_number") or None
                d = meta.get("distance")
                if d is not None:
                    ctx["distance"] = str(d)
                g = meta.get("grade")
                if g is not None:
                    ctx["grade"] = str(g)
        except Exception:
            pass
        # Participants via pandas if available, else fallback to simple parse
        try:
            import pandas as pd  # type: ignore

            try:
                df = pd.read_csv(race_file_path)
            except Exception:
                df = None
            if df is not None and "Dog Name" in df.columns:
                INVALID = {
                    "nan",
                    "none",
                    "null",
                    "n/a",
                    "na",
                    "unknown",
                    "unnamed",
                    "placeholder",
                    "tba",
                    "tbd",
                    "",
                }
                # Build (box_int, name) entries
                entries: list[tuple[Optional[int], str]] = []
                has_box_col = "Box" in df.columns
                # Iterate rows (limited)
                for idx, row in df.iterrows():
                    try:
                        name_raw = str(row.get("Dog Name", "")).strip().strip('"')
                    except Exception:
                        name_raw = ""
                    if not name_raw:
                        continue
                    box_int: Optional[int] = None
                    # If Box column exists, try to parse it first
                    if has_box_col:
                        try:
                            box_val = row.get("Box")
                            if box_val is not None and str(box_val).strip() != "":
                                bi = int(float(str(box_val)))
                                if 1 <= bi <= 8:
                                    box_int = bi
                        except Exception:
                            box_int = None
                    # If box still unknown, try to parse numeric prefix from name
                    parts = name_raw.split(". ", 1)
                    if box_int is None and len(parts) == 2 and parts[0].isdigit():
                        try:
                            bi = int(parts[0])
                            if 1 <= bi <= 8:
                                box_int = bi
                            # Clean name to the suffix
                            name_clean = parts[1].strip()
                        except Exception:
                            name_clean = name_raw
                    else:
                        name_clean = name_raw
                    ls = name_clean.strip().lower()
                    if ls in INVALID or len(name_clean) < 2 or name_clean.isdigit():
                        continue
                    entries.append((box_int, name_clean))
                    if len(entries) >= 24:
                        break
                # Deduplicate by box then name; prefer rows with a valid box
                seen_boxes = set()
                seen_names = set()
                dedup: list[tuple[Optional[int], str]] = []
                for bi, nm in entries:
                    key_nm = nm.strip().lower()
                    if bi is not None:
                        if bi in seen_boxes:
                            continue
                        seen_boxes.add(bi)
                    if key_nm in seen_names:
                        continue
                    seen_names.add(key_nm)
                    dedup.append((bi, nm))
                # Order by box if available, else preserve input order
                dedup.sort(key=lambda x: (999 if x[0] is None else x[0], x[1]))
                # Format as display strings with box prefix when available
                formatted = [
                    (f"B{bi} {nm}" if bi is not None else nm) for bi, nm in dedup
                ]
                ctx["participants"] = formatted[:8]
        except Exception:
            # Fallback: line-based parse
            try:
                entries: list[tuple[Optional[int], str]] = []
                with open(race_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    INVALID = {
                        "nan",
                        "none",
                        "null",
                        "n/a",
                        "na",
                        "unknown",
                        "unnamed",
                        "placeholder",
                        "tba",
                        "tbd",
                        "",
                    }
                    for line in f:
                        line = line.strip().strip('"')
                        if not line or "," not in line:
                            continue
                        # Use first field as Dog Name surrogate
                        first = line.split(",", 1)[0].strip()
                        parts = first.split(". ", 1)
                        box_int: Optional[int] = None
                        name_clean = None
                        if len(parts) == 2 and parts[0].isdigit():
                            try:
                                bi = int(parts[0])
                                if 1 <= bi <= 8:
                                    box_int = bi
                            except Exception:
                                box_int = None
                            name_clean = parts[1].strip()
                        else:
                            name_clean = first
                        ls = (name_clean or "").strip().lower()
                        if not name_clean or name_clean.isdigit() or ls in INVALID:
                            continue
                        entries.append((box_int, name_clean))
                        if len(entries) >= 24:
                            break
                # Deduplicate and order
                seen_boxes = set()
                seen_names = set()
                dedup: list[tuple[Optional[int], str]] = []
                for bi, nm in entries:
                    key_nm = nm.strip().lower()
                    if bi is not None:
                        if bi in seen_boxes:
                            continue
                        seen_boxes.add(bi)
                    if key_nm in seen_names:
                        continue
                    seen_names.add(key_nm)
                    dedup.append((bi, nm))
                dedup.sort(key=lambda x: (999 if x[0] is None else x[0], x[1]))
                formatted = [
                    (f"B{bi} {nm}" if bi is not None else nm) for bi, nm in dedup
                ]
                ctx["participants"] = formatted[:8]
            except Exception:
                pass
        return ctx

    def enhance_race_prediction(
        self,
        race_file_path: str,
        include_betting_strategy: bool = True,
        include_pattern_analysis: bool = True,
    ) -> Dict[str, Any]:
        if not self.gpt_available:
            return {"error": self._error or "OpenAI unavailable"}

        filename = os.path.basename(race_file_path or "")
        # Collect lightweight CSV context to enrich prompt
        csv_ctx = self._collect_csv_context(race_file_path)
        participants = ", ".join(csv_ctx.get("participants") or [])
        venue = csv_ctx.get("venue") or "Unknown"
        race_date = csv_ctx.get("race_date") or "Unknown"
        race_number = csv_ctx.get("race_number") or "Unknown"
        distance = csv_ctx.get("distance") or "Unknown"
        grade = csv_ctx.get("grade") or "Unknown"

        system = (
            "You are a concise greyhound racing analyst. \n"
            "Output strict JSON. Do not include any explanation outside JSON."
        )
        prompt = f"""
Produce a JSON object with keys:
- race_info: include filename as 'filename', and echo venue, race_date, race_number, distance, grade when provided
- gpt_race_analysis: include keys analysis_confidence (0.0-1.0) and summary (string)
- enhanced_ml_predictions: include enhanced_insights as an array of short strings
- betting_strategy: include betting_strategy (string) if requested, else empty
- pattern_analysis: include venue_patterns (string) if requested, else empty
- merged_predictions: an array (can be empty); DO NOT fabricate probabilities; leave empty if no ML context
- analysis_summary: include keys gpt_available (true), prediction_enhancement (true)
Also include top-level timestamp (ISO8601).

Context:
- Race file name: {filename}
- CSV metadata: venue={venue}, date={race_date}, race_number={race_number}, distance={distance}, grade={grade}
- Participants (up to 8): {participants if participants else 'N/A'}
- Only use provided context; do not invent winners or results.
- Keep strings short and actionable.
- analysis_confidence should reflect context richness (use lower values when limited context).
- If betting/pattern inputs are disabled, keep their sections present but minimal.
- If betting_strategy is requested, provide a simple staking or selection note referencing 1-2 likely candidates and risk notes.
- If pattern_analysis is requested, add 1-2 concise venue or box pattern observations, based only on venue/distance/grade when participants are limited.
"""
        data = self._respond_json_with_usage(prompt=prompt, system=system)
        if "error" in data:
            return data
        # Post-process: ensure required shape and add tokens_used to top-level
        tokens_used = (data.get("_meta") or {}).get("tokens_used")
        data.pop("_meta", None)

        # Coerce container types to avoid type errors from model variations
        if not isinstance(data.get("race_info"), dict):
            data["race_info"] = {}
        if not isinstance(data.get("gpt_race_analysis"), dict):
            data["gpt_race_analysis"] = {}
        if not isinstance(data.get("analysis_summary"), dict):
            data["analysis_summary"] = {}
        if not isinstance(data.get("merged_predictions"), list):
            data["merged_predictions"] = []
        # Normalize optional sections that may arrive as strings
        bs = data.get("betting_strategy")
        if not isinstance(bs, dict):
            data["betting_strategy"] = {
                "betting_strategy": bs if isinstance(bs, str) else ""
            }
        pa = data.get("pattern_analysis")
        if not isinstance(pa, dict):
            data["pattern_analysis"] = {
                "venue_patterns": pa if isinstance(pa, str) else ""
            }

        # Minimal required fields
        ri = data["race_info"]
        ri.setdefault("filename", filename)
        # Try to backfill race_info with csv_ctx when present
        if venue and (ri.get("venue") in (None, "", "Unknown")):
            ri["venue"] = venue
        if race_date and (ri.get("date") in (None, "", "Unknown")):
            ri["date"] = race_date
        if race_number and (ri.get("race_number") in (None, "", "Unknown")):
            ri["race_number"] = race_number
        if distance and (ri.get("distance") in (None, "", "Unknown")):
            ri["distance"] = distance
        if grade and (ri.get("grade") in (None, "", "Unknown")):
            ri["grade"] = grade

        data["gpt_race_analysis"].setdefault("analysis_confidence", 0.4)
        data["analysis_summary"].setdefault("gpt_available", True)
        data["timestamp"] = datetime.now().isoformat()
        if tokens_used is not None:
            data["tokens_used"] = tokens_used

        # Respect toggles (force minimal structures regardless of model output type)
        if not include_betting_strategy:
            data["betting_strategy"] = {"betting_strategy": ""}
        if not include_pattern_analysis:
            data["pattern_analysis"] = {"venue_patterns": ""}

        # Fallbacks: if toggles are enabled but sections are empty, provide concise templated content
        try:
            if include_betting_strategy:
                bs_obj = data.get("betting_strategy")
                if not isinstance(bs_obj, dict):
                    bs_obj = {}
                    data["betting_strategy"] = bs_obj
                bs_text = (bs_obj.get("betting_strategy") or "").strip()
                if not bs_text:
                    picks = csv_ctx.get("participants") or []
                    if isinstance(picks, list):
                        picks = [p for p in picks if isinstance(p, str) and p.strip()]
                    top = ", ".join(picks[:2]) if picks else "top selection"
                    bs_obj["betting_strategy"] = (
                        f"Small win bet on {top}. Conservative staking (0.5–1u). Manage risk if wide boxes dominate early."
                    )
            if include_pattern_analysis:
                pa_obj = data.get("pattern_analysis")
                if not isinstance(pa_obj, dict):
                    pa_obj = {}
                    data["pattern_analysis"] = pa_obj
                pa_text = (pa_obj.get("venue_patterns") or "").strip()
                if not pa_text:
                    v = venue if isinstance(venue, str) else "this track"
                    dist_s = str(distance) if distance is not None else "distance"
                    if dist_s.isdigit():
                        dist_s = f"{dist_s}m"
                    g = grade if isinstance(grade, str) else ""
                    g_part = f", {g} grade" if g else ""
                    pa_obj["venue_patterns"] = (
                        f"{v} {dist_s}{g_part}: inside draws and early speed often help. Watch the first split and adjust expectations for track bias."
                    )
        except Exception:
            # Never fail enhancement on fallback generation
            pass

        return data

    def enhance_multiple_races(
        self, race_files: List[str], max_races: int = 5
    ) -> Dict[str, Any]:
        if not isinstance(race_files, list):
            return {"error": "race_files must be a list"}
        limited = race_files[: max(0, int(max_races or 0)) or len(race_files)]
        successes: List[Dict[str, Any]] = []
        failures: List[str] = []
        total_tokens = 0
        for rf in limited:
            try:
                res = self.enhance_race_prediction(rf)
                if "error" in res:
                    failures.append(rf)
                else:
                    total_tokens += int(res.get("tokens_used") or 0)
                    successes.append({"race_file": rf, "enhancement": res})
            except Exception:
                failures.append(rf)
        estimated_cost = (total_tokens or 0) * 0.045 / 1000.0
        return {
            "successful_enhancements": successes,
            "failed": failures,
            "batch_summary": {
                "successful_enhancements": len(successes),
                "failed": len(failures),
                "total_tokens_used": int(total_tokens),
                "estimated_cost_usd": float(estimated_cost),
            },
        }

    def generate_daily_insights(self, date_str: str) -> Dict[str, Any]:
        if not self.gpt_available:
            return {"date": date_str, "insights": [], "summary": "GPT unavailable"}
        system = "You summarize daily racing context. Output strict JSON only."
        prompt = f"""
Return JSON with keys:
- date: echo input date
- insights: array of 3-6 short bullet strings
- summary: one sentence summary
Date: {date_str}
No winners or results should be fabricated.
"""
        data = self._respond_json_with_usage(prompt=prompt, system=system)
        if "error" in data:
            return {"date": date_str, "insights": [], "summary": str(data["error"])}
        # Fallbacks to ensure minimal content
        try:
            data.setdefault("date", date_str)
            ins = data.get("insights")
            if not isinstance(ins, list):
                ins = []
            # Coerce to strings and drop empties
            ins = [str(x).strip() for x in ins if str(x).strip()]
            while len(ins) < 3:
                if len(ins) == 0:
                    ins.append(
                        "Track conditions and early pace will influence outcomes today."
                    )
                elif len(ins) == 1:
                    ins.append(
                        "Monitor inside draws and late market changes for signals."
                    )
                else:
                    ins.append("Use conservative staking on higher-confidence setups.")
            data["insights"] = ins[:6]
            summ = data.get("summary")
            if not isinstance(summ, str) or not summ.strip():
                data["summary"] = (
                    f"{date_str}: Stable conditions and competitive fields; watch early pace and track bias."
                )
        except Exception:
            pass
        return data

    def create_comprehensive_report(self, race_ids: List[str]) -> Dict[str, Any]:
        if not self.gpt_available:
            return {
                "title": "Comprehensive Report (Degraded)",
                "executive_summary": "GPT unavailable. This is a placeholder report.",
                "detailed_analysis": "No analysis available.",
                "recommendations": [],
            }
        race_ids = race_ids or []
        system = "You produce structured racing reports. Output strict JSON only."
        prompt = f"""
Create a JSON report with keys:
- title (string)
- executive_summary (string, 2-4 sentences)
- detailed_analysis (string, concise)
- recommendations (array of 3-5 short items)
Constraints:
- Do not invent winners or results.
- Use only race_ids as labels/context.
Race IDs: {', '.join(race_ids)}
"""
        data = self._respond_json_with_usage(prompt=prompt, system=system)
        if "error" in data:
            return {
                "title": "Comprehensive Report (Error)",
                "executive_summary": str(data["error"]),
                "detailed_analysis": "",
                "recommendations": [],
            }
        # Minimal shape guarantees
        data.setdefault("title", "Comprehensive Report")
        data.setdefault("executive_summary", "")
        data.setdefault("detailed_analysis", "")
        data.setdefault("recommendations", [])
        # Fallback content to ensure readability
        try:
            if (
                not isinstance(data.get("executive_summary"), str)
                or not data.get("executive_summary").strip()
            ):
                n = len(race_ids)
                races_label = f"{n} race(s)" if n else "the selected races"
                data["executive_summary"] = (
                    f"Summary generated from {races_label}. Focus on early speed, inside draws, and recent form indicators. Manage staking conservatively."
                )
            recs = data.get("recommendations")
            if not isinstance(recs, list) or len(recs) == 0:
                data["recommendations"] = [
                    "Monitor early pace and inside draws; adjust expectations for track bias",
                    "Use conservative staking (0.5–1u) on higher confidence picks",
                    "Watch late market/track changes and reassess risk",
                ]
        except Exception:
            pass
        return data
