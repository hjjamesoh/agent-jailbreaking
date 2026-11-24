"""Small, explicitly exploratory CAST response-scope pilot. No new vector training."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cast_stage1_v1"))
import data as shared  # noqa: E402
from core import (Decode, Steering, REFUSAL_PHRASES, REFUSAL_SOURCE, diagnostics,  # noqa: E402
                  format_ids, mixed_prompt, refusal_score, segment)

SCREEN_INDICES = [0, 6, 13, 16]
ARMS = ["baseline", "baseline_replay", "global_response", "local_A", "local_A_replay"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(run):
    return [read(p) for p in sorted((run / "records").glob("*.json"))]


def assets(stage1, count):
    prompts = shared.validate_prepared(stage1)
    bank = read(stage1 / "refusal.svec")
    manifest = read(stage1 / "direction_manifest.json")
    if shared.sha_file(stage1 / "refusal.svec") != manifest["svec_sha256"]:
        raise ValueError("Stage 1 direction checksum mismatch")
    harmful = sorted((r for r in prompts if r["kind"] == "harmful"), key=lambda r: r["id"])
    benign = sorted((r for r in prompts if r["kind"] == "benign"), key=lambda r: r["id"])
    if len(harmful) != 20 or len(benign) != 20:
        raise ValueError("Expected original stage 1's 20+20 calibration prompts")
    screen = [(harmful[i], benign[i]) for i in SCREEN_INDICES]
    remaining = [(h, b) for i, (h, b) in enumerate(zip(harmful, benign))
                 if i not in SCREEN_INDICES]
    if not 1 <= count <= len(remaining):
        raise ValueError(f"--pairs must be 1..{len(remaining)}")
    return screen, remaining[:count], {int(k): v for k, v in bank["directions"].items()}, manifest


def new_run(args, stage, manifest):
    settings = {
        "stage": stage, "stage1": str(args.stage1), "pairs": args.pairs,
        "alpha": args.alpha, "screen_alphas": args.screen_alphas,
        "a_max_tokens": args.a_max_tokens, "b_max_tokens": args.b_max_tokens,
        "direction_sha256": manifest["svec_sha256"],
        "stage1_prompts_sha256": shared.sha_file(args.stage1 / "prompts.json"),
        "model_fingerprint": manifest["fingerprint"], "layers": shared.LAYERS,
        "scope": "No prompt-prefill steering. First generated A token uses unsteered logits. "
                 "Selected A tokens are processed under steering; B boundary is forced. "
                 "Local_A disables ALL hooks before consuming the B boundary.",
        "generation": "greedy; no repetition penalty; explicit EOS union; one batch",
        "screen_indices": SCREEN_INDICES,
        "refusal_rule": {"source": REFUSAL_SOURCE, "substrings": REFUSAL_PHRASES},
        "code_sha256": shared.digest({p.name: shared.sha_file(p) for p in HERE.glob("*.py")}),
        "qualification": "Exploratory forced-boundary diagnostic; no claim of CAST reproduction, "
                         "natural mixed-output refusal, novelty, or harmfulness/ASR evaluation.",
    }
    run = args.output / (stage + "_" + shared.digest(settings)[:12])
    (run / "records").mkdir(parents=True, exist_ok=True)
    target = run / "settings.json"
    if target.exists() and read(target) != settings:
        raise ValueError("Settings collision")
    shared.write_json(target, settings)
    (args.output / f"latest_{stage}.txt").write_text(str(run), encoding="utf-8")
    return run


def enrich(section, expected=None):
    result = {**section, "substring_refusal": refusal_score(section["text"]),
              **diagnostics(section["text"])}
    if expected is not None:
        value = shared.final_number(section["text"])
        result["answer_format_valid"] = value is not None
        result["numeric_correct"] = value is not None and value == shared.decimal_value(expected)
        # Keep correctness and truncation separate; do not call format failure over-refusal.
    return result


def screen(args, model, tok, directions, run, pairs):
    total = len(pairs) * 2 * len(args.screen_alphas)
    for alpha in args.screen_alphas:
        steering = Steering(model, directions, shared.LAYERS, alpha)
        try:
            for h, b in pairs:
                for row in [h, b]:
                    key = shared.digest([alpha, row["id"]])[:20]
                    target = run / "records" / f"{key}.json"
                    if target.exists():
                        continue
                    engine = Decode(model, tok, steering)
                    engine.feed(format_ids(tok, row["prompt"]), False)
                    result = engine.generate(args.b_max_tokens, alpha != 0)
                    shared.write_json(target, {
                        "key": key, "stage": "screen", "alpha": alpha,
                        "prompt_id": row["id"], "kind": row["kind"], "prompt": row["prompt"],
                        "expected": row["expected"],
                        **enrich(result, row["expected"] if row["kind"] == "benign" else None),
                    })
                    print(f"SCREEN {len(list((run / 'records').glob('*.json')))}/{total} "
                          f"alpha={alpha} {row['id']}", flush=True)
        finally:
            steering.close()
    shared.write_json(run / "COMPLETE.json", {"expected": total, "completed": len(records(run))})


def pilot(args, model, tok, directions, run, pairs):
    steering = Steering(model, directions, shared.LAYERS, args.alpha)
    total = len(pairs) * len(ARMS)
    try:
        for index, (h, b) in enumerate(pairs):
            targets = {arm: run / "records" / f"pair_{index:03d}_{arm}.json" for arm in ARMS}
            if all(p.exists() for p in targets.values()) and (
                run / "parity" / f"pair_{index:03d}.json"
            ).exists():
                continue
            prompt = mixed_prompt(h["original_prompt"], b["original_prompt"])
            ids = format_ids(tok, prompt, "<BEGIN_A>\n")
            references = {}
            boundary_logits = {}
            # An interrupted pair is recomputed as a unit; previous complete pairs are retained.
            for arm in ARMS:
                a_enabled = arm.startswith("local_A") or arm == "global_response"
                group = "steered_A" if a_enabled else "unsteered_A"
                result = segment(model, tok, steering, ids, args.a_max_tokens, args.b_max_tokens,
                                 a_enabled, arm == "global_response", arm.endswith("replay"),
                                 references.get(group))
                references[group] = result["a"]["token_ids"]
                boundary_logits[arm] = result.pop("boundary_logits")
                key = targets[arm].stem
                shared.write_json(targets[arm], {
                    "key": key, "stage": "pilot", "pair": index, "arm": arm,
                    "harmful_id": h["id"], "benign_id": b["id"],
                    "harmful_prompt": h["original_prompt"], "benign_prompt": b["original_prompt"],
                    "mixed_prompt": prompt, "expected": b["expected"],
                    **{k: v for k, v in result.items() if k not in ("a", "b")},
                    "a": enrich(result["a"]), "b": enrich(result["b"], b["expected"]),
                    "forced_continuation_after_A_eos": result["a"]["stop_reason"] == "eos",
                })
                print(f"PILOT pair={index + 1}/{len(pairs)} arm={arm}", flush=True)
            rows = {arm: read(path) for arm, path in targets.items()}
            shared.write_json(run / "parity" / f"pair_{index:03d}.json", {
                "pair": index,
                "baseline_replay_B_token_exact": rows["baseline"]["b"]["token_ids"]
                == rows["baseline_replay"]["b"]["token_ids"],
                "baseline_boundary_logit_max_abs_difference": float(
                    (boundary_logits["baseline"] - boundary_logits["baseline_replay"]).abs().max()),
                "local_replay_boundary_logit_max_abs_difference": float(
                    (boundary_logits["local_A"] - boundary_logits["local_A_replay"]).abs().max()),
            })
    finally:
        steering.close()
    if len(records(run)) != total or len(list((run / "parity").glob("*.json"))) != len(pairs):
        raise RuntimeError("Incomplete pilot; rerun the same command")
    shared.write_json(run / "COMPLETE.json", {"expected": total, "completed": len(records(run))})


def report(run):
    rows = records(run)
    groups = defaultdict(list)
    stage = read(run / "settings.json")["stage"]
    for r in rows:
        key = f"alpha={r['alpha']} | {r['kind']}" if stage == "screen" else r["arm"]
        groups[key].append(r)
    summary = {}
    for key, group in groups.items():
        answers = group if stage == "screen" else [r["b"] for r in group]
        a = group if stage == "screen" else [r["a"] for r in group]
        summary[key] = {
            "n": len(group), "A_or_screen_substring_refusals": sum(r["substring_refusal"] for r in a),
            "B_or_screen_numeric_correct": sum(r.get("numeric_correct", False) for r in answers),
            "B_or_screen_numeric_evaluated": sum("numeric_correct" in r for r in answers),
            "B_or_screen_truncated": sum(r["truncated"] for r in answers),
            "A_or_screen_max_word_run": max(r["max_identical_word_run"] for r in a),
            "A_or_screen_fabricated_turns": sum(r["fabricated_turn_marker"] for r in a),
        }
    completed = read(run / "COMPLETE.json") if (run / "COMPLETE.json").exists() else {}
    result = {"stage": stage, "complete": bool(completed) and len(rows) == completed["expected"],
              "groups": summary,
              "warning": "Substring refusal is not safety. No new composite safety score. "
                         "All automated repetition diagnostics are exploratory. "
                         "A caps/EOS and baseline replay parity must be reviewed before causal claims."}
    if stage == "pilot":
        lookup = {(r["pair"], r["arm"]): r for r in rows}
        paired = []
        for pair in sorted({r["pair"] for r in rows}):
            if any((pair, arm) not in lookup for arm in ARMS):
                continue
            rs = {arm: lookup[pair, arm] for arm in ARMS}
            parity_path = run / "parity" / f"pair_{pair:03d}.json"
            parity = read(parity_path) if parity_path.exists() else {}
            paired.append({
                "pair": pair,
                "correct": {arm: r["b"]["numeric_correct"] for arm, r in rs.items()},
                "baseline_to_local_lost": rs["baseline"]["b"]["numeric_correct"]
                and not rs["local_A"]["b"]["numeric_correct"],
                "local_to_replay_recovered": not rs["local_A"]["b"]["numeric_correct"]
                and rs["local_A_replay"]["b"]["numeric_correct"],
                **parity,
            })
        result["paired"] = paired
        result["complete"] = result["complete"] and all(
            "baseline_replay_B_token_exact" in r for r in paired)
        result["baseline_replay_token_mismatches"] = sum(
            r.get("baseline_replay_B_token_exact") is False for r in paired)
        result["baseline_to_local_losses"] = sum(r["baseline_to_local_lost"] for r in paired)
        result["local_to_replay_recoveries"] = sum(r["local_to_replay_recovered"] for r in paired)
    shared.write_json(run / "summary.json", result)
    with (run / "responses.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["screen", "run", "all", "report", "status"])
    parser.add_argument("--stage1", type=Path, default=Path("runs/cast_stage1_v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/cast_scope_v1"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--screen-alphas", type=float, nargs="+", default=[0, 0.1, 0.2, 0.4, 0.85])
    parser.add_argument("--a-max-tokens", type=int, default=128)
    parser.add_argument("--b-max-tokens", type=int, default=256)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    if not args.gpu.isdigit():
        parser.error("--gpu must be a single physical GPU index")
    if args.alpha <= 0 or any(x < 0 for x in args.screen_alphas):
        parser.error("Use positive pilot alpha and nonnegative screen alphas")
    if args.a_max_tokens < 1 or args.b_max_tokens < 1:
        parser.error("Token limits must be positive")
    args.stage1, args.output = args.stage1.resolve(), args.output.resolve()
    if args.command in ["report", "status"]:
        runs = [args.run_dir.resolve()] if args.run_dir else sorted(
            p.parent for p in args.output.glob("*/settings.json"))
        for run in runs:
            if args.command == "report":
                report(run)
            else:
                print(run, "records=", len(records(run)), "complete=", (run / "COMPLETE.json").exists())
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    screen_pairs, pilot_pairs, directions, manifest = assets(args.stage1, args.pairs)
    # Imports torch only AFTER selecting the physical GPU.
    from pipeline import load_model
    model, tok, fingerprint = load_model(args.stage1, args.local_only)
    if fingerprint != manifest["fingerprint"]:
        raise RuntimeError("Model/tokenizer/environment differs from vector extraction. "
                           "Use the same stage 1 Python environment.")
    for stage in (["screen", "pilot"] if args.command == "all" else
                  ["pilot" if args.command == "run" else "screen"]):
        run = new_run(args, stage, manifest)
        print(f"RUN_DIR={run}", flush=True)
        if stage == "screen":
            screen(args, model, tok, directions, run, screen_pairs)
        else:
            pilot(args, model, tok, directions, run, pilot_pairs)
        report(run)


if __name__ == "__main__":
    main()
