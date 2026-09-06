import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from refusal_repro.agent_env import format_agent_context, run_agent_task, write_jsonl
from refusal_repro.data import load_jsonl
from refusal_repro.directions import compute_candidate_directions, mean_residuals_from_texts
from refusal_repro.metrics import (
    average_refusal_metric_from_texts,
    refusal_token_ids,
    summarize_completion_refusals,
)
from refusal_repro.modeling import get_decoder_layers, load_model_and_tokenizer, runtime_metadata
from refusal_repro.selection import resolve_candidate_layers, select_best_candidate


def parse_args():
    p = argparse.ArgumentParser(
        description="Experiment 1: test refusal directions in a simple agent loop."
    )
    p.add_argument("--mode", choices=["detect", "apply", "both"], default="both")
    p.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    p.add_argument(
        "--harmful-train",
        default="experiments/exp0_llm_refusal_dir/data/harmful_train.jsonl",
    )
    p.add_argument(
        "--harmless-train",
        default="experiments/exp0_llm_refusal_dir/data/harmless_train.jsonl",
    )
    p.add_argument(
        "--harmful-val",
        default="experiments/exp0_llm_refusal_dir/data/harmful_val.jsonl",
    )
    p.add_argument(
        "--harmless-val",
        default="experiments/exp0_llm_refusal_dir/data/harmless_val.jsonl",
    )
    p.add_argument("--exp0-direction", default=None)
    p.add_argument("--out", default="runs/exp1_agent_refusal_dir")
    p.add_argument("--run-name", default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--max-agent-steps", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--agent-eval-examples", type=int, default=8)
    p.add_argument("--positions", type=int, nargs="+", default=[-1, -2, -3, -4, -5])
    p.add_argument("--candidate-layers", type=int, nargs="*", default=None)
    p.add_argument("--prune-layer-percentage", type=float, default=0.2)
    p.add_argument("--ablation-alpha", type=float, default=1.0)
    p.add_argument("--addition-alpha", type=float, default=1.0)
    p.add_argument(
        "--intervention-modes",
        nargs="+",
        choices=["baseline", "subtract", "add"],
        default=["baseline", "subtract", "add"],
    )
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    return p.parse_args()


def _repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _git_commit_hash():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _load_splits(args):
    harmful_train = load_jsonl(_repo_path(args.harmful_train))
    harmless_train = load_jsonl(_repo_path(args.harmless_train))
    harmful_val = load_jsonl(_repo_path(args.harmful_val))
    harmless_val = load_jsonl(_repo_path(args.harmless_val))
    if args.limit_train is not None:
        harmful_train = harmful_train[:args.limit_train]
        harmless_train = harmless_train[:args.limit_train]
    if args.limit_val is not None:
        harmful_val = harmful_val[:args.limit_val]
        harmless_val = harmless_val[:args.limit_val]
    return harmful_train, harmless_train, harmful_val, harmless_val


def _agent_contexts(tokenizer, prompts):
    return [format_agent_context(tokenizer, prompt) for prompt in prompts]


def _token_audit_for_texts(tokenizer, dataset_name, texts, positions, max_length, max_examples=3):
    rows = []
    for prompt_index, text in enumerate(texts[:max_examples]):
        full_ids = tokenizer(text, add_special_tokens=False, truncation=False).input_ids
        used_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        ).input_ids
        token_rows = []
        for position in positions:
            absolute_index = len(used_ids) + position if position < 0 else position
            in_range = 0 <= absolute_index < len(used_ids)
            token_id = int(used_ids[absolute_index]) if in_range else None
            token_rows.append({
                "relative_position": int(position),
                "index_in_truncated_unpadded_sequence": int(absolute_index),
                "in_range": bool(in_range),
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]) if token_id is not None else None,
            })
        rows.append({
            "dataset": dataset_name,
            "prompt_index": prompt_index,
            "formatted_agent_context": text,
            "num_tokens_before_truncation": len(full_ids),
            "num_tokens_after_truncation": len(used_ids),
            "truncated": len(full_ids) > len(used_ids),
            "positions": token_rows,
        })
    return rows


def _load_direction(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if "direction" not in payload:
        raise ValueError(f"{path} does not contain a selected 'direction' tensor.")
    metrics = payload.get("metrics", {})
    layer = payload.get("layer", metrics.get("layer"))
    position = payload.get("position", metrics.get("position"))
    if layer is None:
        raise ValueError(f"{path} does not record a selected layer.")
    return {
        "path": str(path),
        "payload": payload,
        "direction": payload["direction"],
        "layer": int(layer),
        "position": int(position) if position is not None else None,
        "metrics": metrics,
    }


def _summarize_agent_rows(rows):
    if not rows:
        return {
            "num_examples": 0,
            "num_refusals": 0,
            "refusal_rate": None,
            "label_counts": {},
            "first_action_counts": {},
            "num_completed_with_finish": 0,
            "completed_with_finish_rate": None,
            "num_invalid_finalization_actions": 0,
            "num_policy_tool_uses": 0,
            "policy_tool_use_rate": None,
            "num_tool_calls": 0,
            "avg_tool_calls": None,
        }
    num_refusals = sum(1 for row in rows if row["looks_like_refusal"])
    num_tool_calls = sum(
        1
        for row in rows
        for step in row["steps"]
        if step["parsed"]["type"] == "tool"
    )
    label_counts = {}
    first_action_counts = {}
    num_invalid_finalization_actions = 0
    for row in rows:
        label = row.get("final_label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
        first_action = row.get("first_action") or "none"
        first_action_counts[first_action] = first_action_counts.get(first_action, 0) + 1
        num_invalid_finalization_actions += sum(
            1
            for step in row["steps"]
            if step["parsed"]["type"] == "invalid_tool"
        )
    num_policy_tool_uses = sum(1 for row in rows if row.get("used_policy_tool"))
    num_completed_with_finish = sum(1 for row in rows if row.get("completed_with_finish"))
    return {
        "num_examples": len(rows),
        "num_refusals": num_refusals,
        "refusal_rate": num_refusals / len(rows),
        "label_counts": label_counts,
        "first_action_counts": first_action_counts,
        "num_completed_with_finish": num_completed_with_finish,
        "completed_with_finish_rate": num_completed_with_finish / len(rows),
        "num_invalid_finalization_actions": num_invalid_finalization_actions,
        "num_policy_tool_uses": num_policy_tool_uses,
        "policy_tool_use_rate": num_policy_tool_uses / len(rows),
        "num_tool_calls": num_tool_calls,
        "avg_tool_calls": num_tool_calls / len(rows),
    }


def _detect_agent_direction(
    args,
    out,
    model,
    tokenizer,
    n_layers,
    harmful_train,
    harmless_train,
    harmful_val,
    harmless_val,
    refusal_ids,
):
    effective_candidate_layers = resolve_candidate_layers(
        n_layers,
        candidate_layers=args.candidate_layers,
        prune_layer_percentage=(
            None if args.candidate_layers is not None else args.prune_layer_percentage
        ),
    )

    harmful_train_texts = _agent_contexts(tokenizer, harmful_train)
    harmless_train_texts = _agent_contexts(tokenizer, harmless_train)
    harmful_val_texts = _agent_contexts(tokenizer, harmful_val)
    harmless_val_texts = _agent_contexts(tokenizer, harmless_val)

    token_audit = []
    token_audit.extend(_token_audit_for_texts(
        tokenizer, "harmful_train_agent", harmful_train_texts, args.positions, args.max_length
    ))
    token_audit.extend(_token_audit_for_texts(
        tokenizer, "harmless_train_agent", harmless_train_texts, args.positions, args.max_length
    ))
    _write_json(out / "agent_token_audit.json", token_audit)

    base_harm, _ = average_refusal_metric_from_texts(
        model,
        tokenizer,
        harmful_val_texts,
        refusal_ids,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    base_safe, _ = average_refusal_metric_from_texts(
        model,
        tokenizer,
        harmless_val_texts,
        refusal_ids,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    harmful_means = mean_residuals_from_texts(
        model,
        tokenizer,
        harmful_train_texts,
        positions=tuple(args.positions),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    harmless_means = mean_residuals_from_texts(
        model,
        tokenizer,
        harmless_train_texts,
        positions=tuple(args.positions),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    candidates = compute_candidate_directions(harmful_means, harmless_means)

    torch.save(
        {
            "model": args.model,
            "positions": args.positions,
            "effective_candidate_layers": effective_candidate_layers,
            "harmful_means": harmful_means,
            "harmless_means": harmless_means,
            "candidates": candidates,
            "context_type": "agent_generation_boundary",
        },
        out / "agent_candidate_directions.pt",
    )

    best, best_direction, _ = select_best_candidate(
        model=model,
        tokenizer=tokenizer,
        candidates=candidates,
        positions=args.positions,
        harmful_val=harmful_val_texts,
        harmless_val=harmless_val_texts,
        refusal_ids=refusal_ids,
        out_csv=out / "agent_selection_metrics.csv",
        batch_size=args.batch_size,
        max_length=args.max_length,
        candidate_layers=effective_candidate_layers,
        ablation_alpha=args.ablation_alpha,
        inputs_are_texts=True,
        baseline_harmful_refusal=base_harm,
        baseline_harmless_refusal=base_safe,
    )

    direction_payload = {
        "model": args.model,
        "layer": int(best["layer"]),
        "position": int(best["position"]),
        "direction": best_direction.cpu(),
        "metrics": best,
        "refusal_token_ids": refusal_ids,
        "effective_candidate_layers": effective_candidate_layers,
        "context_type": "agent_generation_boundary",
    }
    torch.save(direction_payload, out / "agent_direction.pt")
    _write_json(out / "agent_best_direction.json", best)

    return {
        "baseline_harmful_refusal": base_harm,
        "baseline_harmless_refusal": base_safe,
        "selected_direction": best,
        "effective_candidate_layers": effective_candidate_layers,
        "direction_path": str(out / "agent_direction.pt"),
    }


def _run_agent_interventions(args, out, model, tokenizer, harmful_val, harmless_val):
    if args.exp0_direction is None:
        raise ValueError("--exp0-direction is required for --mode apply or --mode both.")
    loaded = _load_direction(_repo_path(args.exp0_direction))
    direction = loaded["direction"]
    layer = loaded["layer"]

    task_sets = {
        "harmful": harmful_val[:args.agent_eval_examples],
        "harmless": harmless_val[:args.agent_eval_examples],
    }
    summaries = {}
    for split_name, tasks in task_sets.items():
        for mode in args.intervention_modes:
            if mode == "baseline":
                intervention = None
                alpha = 1.0
            elif mode == "subtract":
                intervention = "subtract"
                alpha = args.ablation_alpha
            else:
                intervention = "add"
                alpha = args.addition_alpha

            rows = []
            for task in tasks:
                row = run_agent_task(
                    model,
                    tokenizer,
                    task,
                    max_steps=args.max_agent_steps,
                    max_new_tokens=args.max_new_tokens,
                    layer_idx=layer,
                    direction=direction,
                    intervention=intervention,
                    alpha=alpha,
                )
                row.update({
                    "split": split_name,
                    "intervention_mode": mode,
                    "direction_path": loaded["path"],
                    "layer": layer,
                    "position": loaded["position"],
                    "alpha": alpha,
                })
                rows.append(row)

            path = out / "agent_runs" / f"{split_name}_{mode}.jsonl"
            write_jsonl(path, rows)
            summaries[f"{split_name}_{mode}"] = _summarize_agent_rows(rows)

    return {
        "exp0_direction_path": loaded["path"],
        "exp0_direction_layer": layer,
        "exp0_direction_position": loaded["position"],
        "agent_run_summary": summaries,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out = _repo_path(args.out)
    if args.run_name:
        out = out / args.run_name
    out.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["resolved_out"] = str(out)
    _write_json(out / "config.json", config)

    harmful_train, harmless_train, harmful_val, harmless_val = _load_splits(args)
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=args.dtype)
    n_layers = len(get_decoder_layers(model))
    metadata = runtime_metadata(model=model, tokenizer=tokenizer)
    metadata.update({
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_commit_hash(),
        "num_decoder_layers": n_layers,
        "num_harmful_train": len(harmful_train),
        "num_harmless_train": len(harmless_train),
        "num_harmful_val": len(harmful_val),
        "num_harmless_val": len(harmless_val),
        "agent_environment": "simple ReAct-style loop with safe mock tools",
    })
    _write_json(out / "metadata.json", metadata)

    refusal_ids = refusal_token_ids(tokenizer)
    print("CUDA_VISIBLE_DEVICES:", metadata["cuda_visible_devices"])
    print("Visible GPU names:", metadata["visible_gpu_names"])
    print("Model parameter device:", metadata["model_parameter_device"])
    print("Number of decoder layers:", n_layers)
    print("Refusal token IDs:", refusal_ids)

    metrics = {
        "mode": args.mode,
        "model": args.model,
        "ablation_alpha": args.ablation_alpha,
        "addition_alpha": args.addition_alpha,
    }

    if args.mode in {"detect", "both"}:
        metrics["agent_direction_detection"] = _detect_agent_direction(
            args,
            out,
            model,
            tokenizer,
            n_layers,
            harmful_train,
            harmless_train,
            harmful_val,
            harmless_val,
            refusal_ids,
        )

    if args.mode in {"apply", "both"}:
        metrics["exp0_direction_agent_interventions"] = _run_agent_interventions(
            args,
            out,
            model,
            tokenizer,
            harmful_val,
            harmless_val,
        )

    _write_json(out / "metrics.json", metrics)
    print()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved: {out / 'config.json'}")
    print(f"Saved: {out / 'metadata.json'}")
    print(f"Saved: {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
