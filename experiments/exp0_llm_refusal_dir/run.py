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

from refusal_repro.data import load_jsonl
from refusal_repro.modeling import (
    load_model_and_tokenizer,
    get_decoder_layers,
    runtime_metadata,
)
from refusal_repro.directions import (
    mean_residuals,
    compute_candidate_directions,
    token_position_audit,
)
from refusal_repro.metrics import (
    refusal_token_ids,
    average_refusal_metric,
    generate_benign_with_addition,
    generate_completions,
    summarize_completion_refusals,
)
from refusal_repro.selection import resolve_candidate_layers, select_best_candidate
from refusal_repro.analysis import save_projection_plot


def parse_args():
    p = argparse.ArgumentParser(
        description="Experiment 0: detect a refusal direction in a base LLM."
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-8B-Base",
        help="Hugging Face causal LM used for refusal-direction detection.",
    )
    p.add_argument("--harmful-train", default="experiments/exp0_llm_refusal_dir/data/harmful_train.jsonl")
    p.add_argument("--harmless-train", default="experiments/exp0_llm_refusal_dir/data/harmless_train.jsonl")
    p.add_argument("--harmful-val", default="experiments/exp0_llm_refusal_dir/data/harmful_val.jsonl")
    p.add_argument("--harmless-val", default="experiments/exp0_llm_refusal_dir/data/harmless_val.jsonl")
    p.add_argument("--out", default="runs/exp0_llm_refusal_dir")
    p.add_argument(
        "--run-name",
        default=None,
        help="Optional subdirectory under --out, e.g. pilot_001.",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Optional cap per train split for small server pilots.",
    )
    p.add_argument(
        "--limit-val",
        type=int,
        default=None,
        help="Optional cap per validation split for small server pilots.",
    )
    p.add_argument(
        "--token-audit-examples",
        type=int,
        default=3,
        help="Number of examples per split to inspect in token_audit.json.",
    )
    p.add_argument(
        "--completion-eval-examples",
        type=int,
        default=4,
        help="Number of validation examples per split for completion artifacts.",
    )
    p.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=[-1, -2, -3, -4, -5],
        help="Post-instruction positions counted from end of chat template.",
    )
    p.add_argument(
        "--candidate-layers",
        type=int,
        nargs="*",
        default=None,
        help="Optional layer subset for a quick pilot, e.g. 8 10 12 14 16.",
    )
    p.add_argument(
        "--prune-layer-percentage",
        type=float,
        default=0.2,
        help=(
            "When --candidate-layers is omitted, exclude this fraction of final "
            "layers from selection. Use 0.0 to evaluate every layer."
        ),
    )
    p.add_argument(
        "--ablation-alpha",
        type=float,
        default=1.0,
        help="Scale for refusal-direction removal during selection and harmful completion checks.",
    )
    p.add_argument(
        "--addition-alpha",
        type=float,
        default=1.0,
        help="Scale for refusal-direction addition during benign completion checks.",
    )
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    return p.parse_args()

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

def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def main():
    args = parse_args()
    out = _repo_path(args.out)
    if args.run_name:
        out = out / args.run_name
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "model": args.model,
        "harmful_train": args.harmful_train,
        "harmless_train": args.harmless_train,
        "harmful_val": args.harmful_val,
        "harmless_val": args.harmless_val,
        "out": args.out,
        "resolved_out": str(out),
        "run_name": args.run_name,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "positions": args.positions,
        "candidate_layers": args.candidate_layers,
        "prune_layer_percentage": args.prune_layer_percentage,
        "ablation_alpha": args.ablation_alpha,
        "addition_alpha": args.addition_alpha,
        "token_audit_examples": args.token_audit_examples,
        "completion_eval_examples": args.completion_eval_examples,
        "dtype": args.dtype,
    }
    _write_json(out / "config.json", config)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    model, tokenizer = load_model_and_tokenizer(args.model, dtype=args.dtype)
    n_layers = len(get_decoder_layers(model))
    effective_candidate_layers = resolve_candidate_layers(
        n_layers,
        candidate_layers=args.candidate_layers,
        prune_layer_percentage=(
            None if args.candidate_layers is not None else args.prune_layer_percentage
        ),
    )
    config["effective_candidate_layers"] = effective_candidate_layers
    _write_json(out / "config.json", config)

    metadata = runtime_metadata(model=model, tokenizer=tokenizer)
    metadata.update({
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_commit_hash(),
        "num_decoder_layers": n_layers,
        "num_harmful_train": len(harmful_train),
        "num_harmless_train": len(harmless_train),
        "num_harmful_val": len(harmful_val),
        "num_harmless_val": len(harmless_val),
        "effective_candidate_layers": effective_candidate_layers,
        "candidate_layer_policy": (
            "explicit --candidate-layers"
            if args.candidate_layers is not None
            else f"all layers except final {args.prune_layer_percentage:.2%}"
        ),
        "ablation_alpha": args.ablation_alpha,
        "addition_alpha": args.addition_alpha,
        "data_schema": "JSONL; each row is either a JSON string or an object with a non-empty 'prompt' field",
        "activation_capture": {
            "module": "decoder layer forward hook",
            "stream_location": "decoder block output as returned by transformers layer module",
            "token_positions": args.positions,
            "position_reference": "negative positions are relative to each sample's final non-padding token from attention_mask",
        },
    })
    _write_json(out / "metadata.json", metadata)

    token_audit = []
    for dataset_name, prompts in [
        ("harmful_train", harmful_train),
        ("harmless_train", harmless_train),
        ("harmful_val", harmful_val),
        ("harmless_val", harmless_val),
    ]:
        token_audit.extend(token_position_audit(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            prompts=prompts,
            positions=args.positions,
            max_length=args.max_length,
            max_examples=args.token_audit_examples,
        ))
    _write_json(out / "token_audit.json", token_audit)

    print("CUDA available:", metadata["cuda_available"])
    print("CUDA_VISIBLE_DEVICES:", metadata["cuda_visible_devices"])
    print("Visible CUDA devices:", metadata["visible_gpu_count"])
    print("Visible GPU names:", metadata["visible_gpu_names"])
    print("Model parameter device:", metadata["model_parameter_device"])
    print("Model parameter dtype:", metadata["model_parameter_dtype"])

    refusal_ids = refusal_token_ids(tokenizer)
    print("Refusal token IDs:", refusal_ids)
    print("Number of decoder layers:", n_layers)
    print("Effective candidate layers:", effective_candidate_layers)

    base_harm, _ = average_refusal_metric(
        model,
        tokenizer,
        harmful_val,
        refusal_ids,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    base_safe, _ = average_refusal_metric(
        model,
        tokenizer,
        harmless_val,
        refusal_ids,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"Baseline harmful refusal metric:  {base_harm:.4f}")
    print(f"Baseline harmless refusal metric: {base_safe:.4f}")

    harmful_means = mean_residuals(
        model,
        tokenizer,
        harmful_train,
        positions=tuple(args.positions),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    harmless_means = mean_residuals(
        model,
        tokenizer,
        harmless_train,
        positions=tuple(args.positions),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    candidates = compute_candidate_directions(harmful_means, harmless_means)

    torch.save(
        {
            "model": args.model,
            "positions": args.positions,
            "harmful_means": harmful_means,
            "harmless_means": harmless_means,
            "candidates": candidates,
        },
        out / "candidate_directions.pt",
    )

    best, best_direction, all_results = select_best_candidate(
        model=model,
        tokenizer=tokenizer,
        candidates=candidates,
        positions=args.positions,
        harmful_val=harmful_val,
        harmless_val=harmless_val,
        refusal_ids=refusal_ids,
        out_csv=out / "selection_metrics.csv",
        batch_size=args.batch_size,
        max_length=args.max_length,
        candidate_layers=effective_candidate_layers,
        ablation_alpha=args.ablation_alpha,
        baseline_harmful_refusal=base_harm,
        baseline_harmless_refusal=base_safe,
    )

    best_payload = {
        "model": args.model,
        "layer": int(best["layer"]),
        "position": int(best["position"]),
        "direction": best_direction.cpu(),
        "metrics": best,
        "refusal_token_ids": refusal_ids,
        "config": config,
        "metadata": metadata,
        "effective_candidate_layers": effective_candidate_layers,
        "ablation_alpha": args.ablation_alpha,
        "addition_alpha": args.addition_alpha,
    }
    torch.save(best_payload, out / "direction.pt")

    with open(out / "best_direction.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)

    metrics = {
        "baseline_harmful_refusal": base_harm,
        "baseline_harmless_refusal": base_safe,
        "selected_direction": best,
        "effective_candidate_layers": effective_candidate_layers,
        "ablation_alpha": args.ablation_alpha,
        "addition_alpha": args.addition_alpha,
        "selection_metric": (
            "harmful_refusal_delta - max(0, harmless_refusal_delta); "
            "higher is better under the next-token refusal-prefix proxy"
        ),
    }

    best_position_idx = args.positions.index(int(best["position"]))
    save_projection_plot(
        harmful_means,
        harmless_means,
        best_direction,
        best_position_idx,
        out / "projection_by_layer.png",
    )

    # Safe causal sufficiency check:
    # add the selected vector only on harmless prompts and inspect induced refusals.
    benign_examples = harmless_val[: min(8, len(harmless_val))]
    completions = generate_benign_with_addition(
        model,
        tokenizer,
        benign_examples,
        layer_idx=int(best["layer"]),
        direction=best_direction,
        alpha=args.addition_alpha,
        max_new_tokens=args.max_new_tokens,
    )
    with open(out / "benign_activation_addition_examples.jsonl", "w", encoding="utf-8") as f:
        for prompt, completion in zip(benign_examples, completions):
            f.write(json.dumps(
                {"prompt": prompt, "completion": completion},
                ensure_ascii=False
            ) + "\n")

    completion_sets = {
        "harmful_baseline": generate_completions(
            model,
            tokenizer,
            harmful_val[:args.completion_eval_examples],
            max_new_tokens=args.max_new_tokens,
        ),
        "harmful_ablation": generate_completions(
            model,
            tokenizer,
            harmful_val[:args.completion_eval_examples],
            max_new_tokens=args.max_new_tokens,
            layer_idx=int(best["layer"]),
            direction=best_direction,
            intervention="subtract",
            alpha=args.ablation_alpha,
        ),
        "harmless_baseline": generate_completions(
            model,
            tokenizer,
            harmless_val[:args.completion_eval_examples],
            max_new_tokens=args.max_new_tokens,
        ),
        "harmless_activation_addition": generate_completions(
            model,
            tokenizer,
            harmless_val[:args.completion_eval_examples],
            max_new_tokens=args.max_new_tokens,
            layer_idx=int(best["layer"]),
            direction=best_direction,
            intervention="add",
            alpha=args.addition_alpha,
        ),
    }
    completions_dir = out / "completions"
    completions_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in completion_sets.items():
        _write_json(completions_dir / f"{name}.json", rows)
    metrics["completion_refusal_summary"] = {
        name: summarize_completion_refusals(rows)
        for name, rows in completion_sets.items()
    }
    metrics["completion_refusal_classifier"] = (
        "Simple prefix heuristic over the first 200 normalized characters; "
        "use for smoke checks, not publication-grade evaluation."
    )
    _write_json(out / "metrics.json", metrics)

    print()
    print("Selected direction")
    print(json.dumps(best, indent=2))
    print(f"Saved: {out / 'direction.pt'}")
    print(f"Saved: {out / 'config.json'}")
    print(f"Saved: {out / 'metadata.json'}")
    print(f"Saved: {out / 'token_audit.json'}")
    print(f"Saved: {out / 'metrics.json'}")
    print(f"Saved: {out / 'selection_metrics.csv'}")
    print(f"Saved: {out / 'completions'}")
    print(f"Saved: {out / 'projection_by_layer.png'}")
    print()
    print(
        "Note: completion refusal labels use a simple prefix heuristic. "
        "Use Experiment 1 agent interventions before making agent-level claims."
    )


if __name__ == "__main__":
    main()
