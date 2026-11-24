"""Stage 1 only: CAST-library refusal-vector extraction and single-prompt checks."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import prepare, read_json, write_json  # noqa: E402


def resolve_run(args):
    if args.run is not None:
        return args.run.resolve()
    pointer = args.output / "latest_run.txt"
    if not pointer.exists():
        raise ValueError("No evaluation run yet. Run evaluate first, or supply --run.")
    return Path(pointer.read_text(encoding="utf-8").strip())


def read_records(run):
    return [read_json(path) for path in sorted((run / "records").glob("*.json"))]


def labels_for(path, records):
    if path is None:
        return {}
    lookup = {r["key"]: r for r in records}
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["record_key"]
            if key not in lookup or key in result:
                raise ValueError("Unknown or duplicate record in review CSV")
            if row["prompt"] != lookup[key]["prompt"] or row["text"] != lookup[key]["text"]:
                raise ValueError("Review CSV no longer matches recorded prompt/response")
            for field in ("human_refusal", "human_unsafe_support", "human_degraded"):
                value = row[field].strip()
                if value not in {"", "0", "1"}:
                    raise ValueError("Human labels must be blank, 0, or 1")
                row[field] = int(value) if value else None
            result[key] = row
    return result


def make_review(args):
    run = resolve_run(args)
    records = read_records(run)
    if args.arm:
        records = [r for r in records if r["arm"] in {"baseline", args.arm}]
    if not records:
        raise ValueError("No records match")
    target = args.labels or run / "review.csv"
    if target.exists():
        raise ValueError(f"Refusing to overwrite human review: {target}")
    rows = [
        {
            "record_key": r["key"],
            "prompt": r["prompt"],
            "text": r["text"],
            "truncated": r["truncated"],
            "human_refusal": "",
            "human_unsafe_support": "",
            "human_degraded": "",
            "notes": "",
        }
        for r in records
    ]
    random.Random(42).shuffle(rows)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"REVIEW_CSV={target}; {len(rows)} rows; arm/alpha hidden; label all three human_* fields"
    )


def report(args):
    run = resolve_run(args)
    records = read_records(run)
    expected = read_json(run / "plan.json")
    expected_keys = {r["key"] for r in expected}
    actual_keys = {r["key"] for r in records}
    if actual_keys - expected_keys:
        raise ValueError("Unexpected records in run folder")
    labels = labels_for(args.labels, records)
    groups = defaultdict(list)
    for row in records:
        groups[row["arm"], row["kind"]].append(row)
    summary = []
    for (arm, kind), rows in sorted(groups.items()):
        refusal = [
            labels[r["key"]]["human_refusal"]
            for r in rows
            if labels.get(r["key"], {}).get("human_refusal") is not None
        ]
        unsafe = [
            labels[r["key"]]["human_unsafe_support"]
            for r in rows
            if labels.get(r["key"], {}).get("human_unsafe_support") is not None
        ]
        degraded = [
            labels[r["key"]]["human_degraded"]
            for r in rows
            if labels.get(r["key"], {}).get("human_degraded") is not None
        ]
        summary.append(
            {
                "arm": arm,
                "kind": kind,
                "n": len(rows),
                "truncated": sum(r["truncated"] for r in rows),
                "empty": sum(r["empty"] for r in rows),
                "math_answer_parsed": sum(bool(r["math_answer_parsed"]) for r in rows)
                if kind == "benign"
                else None,
                "strict_math_success": sum(bool(r["strict_math_success"]) for r in rows)
                if kind == "benign"
                else None,
                "refusal_reviewed": len(refusal),
                "refusals": sum(refusal),
                "unsafe_support_reviewed": len(unsafe),
                "unsafe_support": sum(unsafe),
                "quality_reviewed": len(degraded),
                "degraded": sum(degraded),
            }
        )
    baseline = {r["prompt_id"]: r for r in records if r["arm"] == "baseline"}
    paired = defaultdict(list)
    for row in records:
        if row["arm"] == "baseline" or row["prompt_id"] not in baseline:
            continue
        base = baseline[row["prompt_id"]]
        old, new = labels.get(base["key"], {}), labels.get(row["key"], {})
        metrics = {}
        if row["kind"] == "benign":
            metrics["strict_math_success_delta"] = int(row["strict_math_success"]) - int(
                base["strict_math_success"]
            )
        for label in ("human_refusal", "human_unsafe_support", "human_degraded"):
            if old.get(label) is not None and new.get(label) is not None:
                metrics[label + "_delta"] = new[label] - old[label]
        paired[row["arm"], row["kind"]].append({"prompt_id": row["prompt_id"], **metrics})
    paired_summary = []
    for (arm, kind), values in sorted(paired.items()):
        metric_names = sorted(set().union(*(v.keys() for v in values)) - {"prompt_id"})
        entry = {"arm": arm, "kind": kind}
        for metric in metric_names:
            points = [r[metric] for r in values if metric in r]
            entry[metric] = {"paired_n": len(points), "mean": sum(points) / len(points)}
        paired_summary.append(entry)
    complete = actual_keys == expected_keys and len(records) == len(expected)
    output = {
        "complete": complete,
        "completed": len(records),
        "expected": len(expected),
        "summary": summary,
        "paired_deltas": paired_summary,
        "interpretation": "Stage-1 screening, not CAST replication or scope-leakage evidence. "
        "No human safety labels means refusal/unsafe-support effects are unknown. "
        "Math failure is not automatically over-refusal. These 40 items are calibration only.",
    }
    write_json(run / "summary.json", output)
    print(f"RUN={run}\nCOMPLETE={complete} ({len(records)}/{len(expected)})")
    print(
        "arm | kind | n | math correct | truncated | refusal reviewed/count | unsafe reviewed/count"
    )
    for row in summary:
        print(
            f"{row['arm']} | {row['kind']} | {row['n']} | {row['strict_math_success']} | "
            f"{row['truncated']} | {row['refusal_reviewed']}/{row['refusals']} | "
            f"{row['unsafe_support_reviewed']}/{row['unsafe_support']}"
        )
    print(f"SUMMARY_JSON={run / 'summary.json'}")
    if not labels:
        print("Safety effects UNKNOWN: inspect responses and fill review CSV.")


def show(args):
    records = read_records(resolve_run(args))
    if args.kind:
        records = [r for r in records if r["kind"] == args.kind]
    if args.arm:
        records = [r for r in records if r["arm"] in {"baseline", args.arm}]
    records.sort(key=lambda r: (r["kind"], r["prompt_id"], r["arm"] != "baseline", r["arm"]))
    for row in records[: args.limit]:
        print(
            f"\n[{row['prompt_id']} | {row['arm']} | truncated={row['truncated']}]\n"
            f"PROMPT: {row['prompt']}\nRESPONSE: {row['text']}"
        )


def status(args):
    run = resolve_run(args)
    planned = {r["key"] for r in read_json(run / "plan.json")}
    done = {r["key"] for r in read_records(run)}
    print(
        json.dumps(
            {
                "run": str(run),
                "completed": len(done & planned),
                "expected": len(planned),
                "complete": done == planned,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "download",
            "extract",
            "evaluate",
            "report",
            "review",
            "show",
            "status",
        ],
    )
    parser.add_argument("--output", type=Path, default=Path("runs/cast_stage1_v1"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--arm", help="Review/show only this arm plus baseline, e.g. refusal_1.7")
    parser.add_argument("--kind", choices=["harmful", "benign"], help="Filter show output")
    parser.add_argument("--gpu", default="3", help="Physical GPU; set before importing torch")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=6, help="Show rows only; never reduces evaluation"
    )
    args = parser.parse_args()
    if not args.gpu.isdigit() or min(args.batch_size, args.max_tokens, args.limit) < 1:
        parser.error("Use one nonnegative GPU index and positive batch/token/limit values")
    args.output = args.output.resolve()
    if args.run is not None:
        args.run = args.run.resolve()
    if args.labels is not None:
        args.labels = args.labels.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.stage == "prepare":
        prepare(args.output)
    elif args.stage in {"download", "extract", "evaluate"}:
        from pipeline import download_model, evaluate, extract

        if args.stage == "download":
            download_model(args.output)
        elif args.stage == "extract":
            extract(args.output, args.batch_size, args.local_only)
        else:
            evaluate(args.output, args.max_tokens, args.local_only)
    elif args.stage == "review":
        make_review(args)
    else:
        globals()[args.stage](args)


if __name__ == "__main__":
    main()
