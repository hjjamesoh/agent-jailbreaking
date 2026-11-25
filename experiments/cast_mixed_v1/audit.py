"""CPU-only post-hoc audit. Preserve all original generations, labels and scores."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

from protocol import sections

FLEX_PATTERN = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
STRICT_PATTERN = r"#### (\-?[0-9\.\,]+)"
IGNORE = [r",", r"\$", r"(?s).*#### ", r"\.$"]
SOURCE = {
    "url": "https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml",
    "yaml_git_blob_sha1": "9266ab1f80af13b228f55f474442eb8db34f8a13",
    "scope": "Same strict-match/flexible-extract regex, group selection and exact-match normalization; "
             "adapted to B section, not a full lm-eval benchmark rerun or semantic math judge.",
}


def extract(text, flexible):
    if text is None:
        return None
    matches = re.findall(FLEX_PATTERN if flexible else STRICT_PATTERN, text)
    if not matches:
        return None
    match = matches[-1 if flexible else 0]
    if isinstance(match, tuple):
        match = next((s for s in match if s), None)
    return match.strip() if match is not None else None


def normalized(text):
    for pattern in IGNORE:
        text = re.sub(pattern, "", text)
    return text.lower().strip()


def rescore(row):
    part, counts = sections(row["text"])
    text, scope = part["B"], "B_section"
    if row["layout"] == "b_only" and counts["B"] == 0:
        # One task is present: its whole answer is unambiguously a B response.
        text, scope = row["text"], "b_only_full_response"
    if row["layout"] == "h_only":
        return {"B_scope": None, "B_flexible_correct": None, "B_harness_strict_correct": None,
                "B_flexible_extracted": None, "B_harness_strict_extracted": None}
    result = {"B_scope": scope if text is not None else "missing_or_duplicate_B_section"}
    for flexible, name in [(False, "harness_strict"), (True, "flexible")]:
        value = extract(text, flexible)
        result[f"B_{name}_extracted"] = value
        result[f"B_{name}_correct"] = value is not None and normalized(value) == normalized(row["expected"])
    return result


def distribution(values):
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values)} if values else {"n": 0}


def audit(root):
    records = []
    groups = defaultdict(list)
    for path in sorted((root / "records").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        new = rescore(row)
        item = {"key": row["key"], "pair": row["pair"], "arm": row["arm"],
                "layout": row["layout"], "expected": row["expected"],
                "original_B_format_constrained_success": row["B_numeric_correct"], **new,
                "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        records.append(item)
        groups[row["arm"], row["layout"]].append((row, item))
    if not records:
        raise ValueError("No original records found")
    summary = []
    for (arm, layout), values in sorted(groups.items()):
        summary.append({"arm": arm, "layout": layout, "n": len(values),
                        "original_format_constrained_success": sum(r["B_numeric_correct"] is True for r, _ in values),
                        "harness_strict_correct": sum(a["B_harness_strict_correct"] is True for _, a in values),
                        "flexible_correct": sum(a["B_flexible_correct"] is True for _, a in values),
                        "flexible_number_found": sum(a["B_flexible_extracted"] is not None for _, a in values),
                        "B_section_missing": sum(a["B_scope"] == "missing_or_duplicate_B_section" for _, a in values),
                        "gate_similarity": distribution([r["condition_similarities"]["7"] for r, _ in values
                                                         if "7" in r["condition_similarities"]])})
    analysis_path = root / "gate_calibration_analysis.json"
    calibration = {}
    if analysis_path.exists():
        original = json.loads(analysis_path.read_text(encoding="utf-8"))
        raw = original["analysis"]["layer_7"]["similarities"]
        calibration = {k: distribution(v) for k, v in raw.items()}
    gate = json.loads((root / "gate.json").read_text(encoding="utf-8"))
    out = root / "audit_lmeval"
    out.mkdir(exist_ok=True)
    payload = {"metric_source": SOURCE, "posthoc": True, "gate": gate,
               "original_calibration_similarity": calibration, "summary": summary,
               "warning": "Original results are preserved. Flexible extraction can select an intermediate "
                          "or irrelevant final number, especially in broken output: review candidate answers. "
                          "It is not semantic correctness and does not change refusal/safety judgments."}
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    print("POST-HOC AUDIT; original outputs/scores unchanged")
    print("arm | layout | n | original format-success | harness strict | flexible | number found | B section missing")
    for r in summary:
        if r["layout"] != "h_only":
            print(f"{r['arm']} | {r['layout']} | {r['n']} | {r['original_format_constrained_success']} | "
                  f"{r['harness_strict_correct']} | {r['flexible_correct']} | "
                  f"{r['flexible_number_found']} | {r['B_section_missing']}")
    print("RAW gate calibration distributions:", json.dumps(calibration))
    print("Calibration gate:", json.dumps(gate))
    print("Saved:", out / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/cast_mixed_v1"))
    args = parser.parse_args()
    audit(args.output.resolve())
