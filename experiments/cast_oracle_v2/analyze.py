"""CPU-only paired diagnostics. No generations, labels or original scores are changed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def transitions(pairs, before, after):
    result = {"recovered": [], "worsened": [], "both_correct": [], "both_wrong": []}
    for pair in pairs:
        a, b = pair[before], pair[after]
        name = {(False, True): "recovered", (True, False): "worsened",
                (True, True): "both_correct", (False, False): "both_wrong"}[bool(a), bool(b)]
        result[name].append(pair["pair"])
    result["net_gain"] = len(result["recovered"]) - len(result["worsened"])
    return result


def labels(root, row):
    result = {"A_refusal": None, "B_refusal": None, "unsafe_support": None}
    for folder in ["rejection_labels", "harm_labels"]:
        path = root / folder / (row["key"] + ".json")
        if not path.exists():
            continue
        label = read(path)
        if label["response_sha256"] != digest(row["text"]):
            raise ValueError(f"Stale judge label: {path}")
        if folder == "harm_labels":
            result["unsafe_support"] = label["unsafe_support"]
        else:
            for section in ["A", "B"]:
                value = label["labels"][section]
                result[section + "_refusal"] = value["cast_refusal"] if value else None
    return result


def diagnose(root, alpha, quiet=False):
    rows = [read(p) for p in sorted((root / "records").glob("*.json"))]
    lookup = {(r["pair"], r["arm"], float(r["alpha"])): r for r in rows}
    ids = sorted(r["pair"] for r in rows if r["arm"] == "oracle_A" and float(r["alpha"]) == alpha)
    if not ids:
        raise ValueError(f"No oracle_A records for alpha={alpha}")
    comparisons, full = [], []
    for pair in ids:
        selected = {arm: lookup[pair, arm, 0.0 if arm in ["B_only", "baseline", "baseline_replay"] else alpha]
                    for arm in ["B_only", "baseline", "baseline_replay", "global", "oracle_A", "oracle_A_replay"]}
        if not (selected["global"]["a"]["token_ids"] == selected["oracle_A"]["a"]["token_ids"]
                == selected["oracle_A_replay"]["a"]["token_ids"]):
            raise ValueError(f"A outputs differ in pair {pair}; matched comparison invalid")
        local = selected["oracle_A"]
        if local["B_intervention_layer_calls"] != 0 or not local.get("A_boundary_valid"):
            raise ValueError(f"Invalid oracle intervention window: pair {pair}")
        scored = {arm: labels(root, row) for arm, row in selected.items()}
        row = {"pair": pair, "gold": local["expected"], "arms": {},
               "baseline_replay_tokens_equal": selected["baseline"]["b"]["token_ids"]
               == selected["baseline_replay"]["b"]["token_ids"],
               "oracle_replay_tokens_equal": local["b"]["token_ids"]
               == selected["oracle_A_replay"]["b"]["token_ids"]}
        for arm, record in selected.items():
            b, a = record["b"], record["a"]
            row[arm] = b["flexible_correct"]
            row["arms"][arm] = {
                "correct": b["flexible_correct"], "extracted": b.get("flexible_extracted"),
                "strict_correct": b["strict_correct"], "B_cap": b["truncated"],
                "B_max_word_run": b["max_identical_word_run"],
                "B_distinct4": b["distinct_4gram_fraction"],
                "B_fabricated_turn": b["fabricated_turn_marker"],
                "A_cap": a["truncated"] if a else None,
                "A_stop": a["stop_reason"] if a else None,
                "A_max_word_run": a["max_identical_word_run"] if a else None,
                **scored[arm],
            }
        comparisons.append(row)
        full.append({"diagnostics": row, "original_records": selected})
    result = {
        "alpha": alpha, "n": len(ids),
        "global_to_oracle": transitions(comparisons, "global", "oracle_A"),
        "baseline_to_oracle": transitions(comparisons, "baseline", "oracle_A"),
        "oracle_to_replay": transitions(comparisons, "oracle_A", "oracle_A_replay"),
        "baseline_replay_token_mismatches": sum(not r["baseline_replay_tokens_equal"] for r in comparisons),
        "oracle_replay_tokens_changed": sum(not r["oracle_replay_tokens_equal"] for r in comparisons),
        "oracle_remaining_errors": [r["pair"] for r in comparisons if not r["oracle_A"]],
        "baseline_failed_despite_B_only_success": [r["pair"] for r in comparisons
                                                   if r["B_only"] and not r["baseline"]],
        "oracle_failed_on_baseline_correct": [r["pair"] for r in comparisons
                                               if r["baseline"] and not r["oracle_A"]],
        "unsafe_global_to_oracle": {
            "new_unsafe": [r["pair"] for r in comparisons
                           if r["arms"]["global"]["unsafe_support"] is False
                           and r["arms"]["oracle_A"]["unsafe_support"] is True],
            "removed_unsafe": [r["pair"] for r in comparisons
                               if r["arms"]["global"]["unsafe_support"] is True
                               and r["arms"]["oracle_A"]["unsafe_support"] is False],
            "unknown": [r["pair"] for r in comparisons
                        if r["arms"]["global"]["unsafe_support"] is None
                        or r["arms"]["oracle_A"]["unsafe_support"] is None],
        },
        "remaining_error_observations": {
            "count": sum(not r["oracle_A"] for r in comparisons),
            "B_cap": sum(not r["oracle_A"] and r["arms"]["oracle_A"]["B_cap"] for r in comparisons),
            "no_numeric_answer_found": sum(not r["oracle_A"] and r["arms"]["oracle_A"]["extracted"] is None
                                           for r in comparisons),
            "B_cast_refusal_flag": sum(not r["oracle_A"] and r["arms"]["oracle_A"]["B_refusal"] is True
                                       for r in comparisons),
            "A_cap": sum(not r["oracle_A"] and r["arms"]["oracle_A"]["A_cap"] for r in comparisons),
        },
        "cases": comparisons,
        "warning": "Paired exploratory observations, not causal attribution or significance. "
                   "Read full answers for arithmetic, refusal and numerical-extraction errors. "
                   "A matched across global/oracle; baseline has different A text and cache. "
                   "Equal total accuracies do not imply equal outcomes per item.",
    }
    out = root / ("analysis_a" + str(alpha).replace(".", "p"))
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "cases.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in full), encoding="utf-8")
    with (out / "cases.txt").open("w", encoding="utf-8") as f:
        for case in full:
            d, selected = case["diagnostics"], case["original_records"]
            f.write(f"\n{'=' * 60}\nPAIR {d['pair']} GOLD {d['gold']}\n")
            f.write("A TASK: " + selected["oracle_A"]["harmful_prompt"] + "\n")
            f.write("B TASK: " + selected["oracle_A"]["benign_prompt"] + "\n")
            for arm in ["B_only", "baseline", "global", "oracle_A", "oracle_A_replay"]:
                r = selected[arm]
                f.write(f"\n[{arm}] " + json.dumps(d["arms"][arm], ensure_ascii=False) + "\n")
                if r["a"]:
                    f.write("A RESPONSE:\n" + r["a"]["text"] + "\n")
                f.write("B RESPONSE:\n" + r["b"]["text"] + "\n")
    if not quiet:
        print(f"alpha={alpha}, n={len(ids)}; no GPU inference")
        for name in ["global_to_oracle", "baseline_to_oracle", "oracle_to_replay"]:
            print(name, json.dumps(result[name]))
        print("unsafe_global_to_oracle", json.dumps(result["unsafe_global_to_oracle"]))
        print("pair | gold | B_only/base/global/oracle/replay (1=correct) | oracle extracted | A unsafe | B refusal | A cap/B cap")
        for r in comparisons:
            d = r["arms"]["oracle_A"]
            bits = "/".join(str(int(r[a])) for a in ["B_only", "baseline", "global", "oracle_A", "oracle_A_replay"])
            print(f"{r['pair']} | {r['gold']} | {bits} | {d['extracted']} | "
                  f"{d['unsafe_support']} | {d['B_refusal']} | {d['A_cap']}/{d['B_cap']}")
        print("Remaining errors:", json.dumps(result["remaining_error_observations"]))
        print("Full original answers to attach:", out / "cases.txt")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/cast_oracle_v2"))
    parser.add_argument("--alpha", type=float, default=0.85)
    args = parser.parse_args()
    diagnose(args.output.resolve(), args.alpha)
