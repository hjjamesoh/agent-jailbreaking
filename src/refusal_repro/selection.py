import csv
import torch
from .metrics import refusal_metric_with_ablation

def resolve_candidate_layers(n_layers, candidate_layers=None, prune_layer_percentage=None):
    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")

    if candidate_layers is not None:
        layers = list(candidate_layers)
    elif prune_layer_percentage is None:
        layers = list(range(n_layers))
    else:
        if prune_layer_percentage < 0.0 or prune_layer_percentage >= 1.0:
            raise ValueError("--prune-layer-percentage must be in [0.0, 1.0).")
        cutoff = int(n_layers * (1.0 - prune_layer_percentage))
        layers = list(range(cutoff))

    if not layers:
        raise ValueError("No candidate layers were selected.")

    for layer in layers:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"Candidate layer {layer} outside [0, {n_layers - 1}]")

    return layers

def select_best_candidate(
    model,
    tokenizer,
    candidates,
    positions,
    harmful_val,
    harmless_val,
    refusal_ids,
    out_csv,
    batch_size=4,
    max_length=1024,
    candidate_layers=None,
    ablation_alpha=1.0,
    baseline_harmful_refusal=None,
    baseline_harmless_refusal=None,
):
    n_layers = candidates.shape[0]
    layers = resolve_candidate_layers(n_layers, candidate_layers=candidate_layers)

    results = []
    best = None
    best_direction = None

    for layer in layers:
        for p_idx, position in enumerate(positions):
            direction = candidates[layer, p_idx]

            harm_abl, _ = refusal_metric_with_ablation(
                model, tokenizer, harmful_val, refusal_ids,
                layer_idx=layer, direction=direction,
                batch_size=batch_size, max_length=max_length,
                alpha=ablation_alpha,
            )
            safe_abl, _ = refusal_metric_with_ablation(
                model, tokenizer, harmless_val, refusal_ids,
                layer_idx=layer, direction=direction,
                batch_size=batch_size, max_length=max_length,
                alpha=ablation_alpha,
            )

            harmful_refusal_delta = (
                baseline_harmful_refusal - harm_abl
                if baseline_harmful_refusal is not None
                else None
            )
            harmless_refusal_delta = (
                safe_abl - baseline_harmless_refusal
                if baseline_harmless_refusal is not None
                else None
            )
            if harmful_refusal_delta is None or harmless_refusal_delta is None:
                score = -harm_abl
            else:
                score = harmful_refusal_delta - max(0.0, harmless_refusal_delta)
            row = {
                "layer": int(layer),
                "position": int(position),
                "baseline_harmful_refusal": (
                    float(baseline_harmful_refusal)
                    if baseline_harmful_refusal is not None
                    else None
                ),
                "baseline_harmless_refusal": (
                    float(baseline_harmless_refusal)
                    if baseline_harmless_refusal is not None
                    else None
                ),
                "harmful_refusal_after_ablation": float(harm_abl),
                "harmless_refusal_after_ablation": float(safe_abl),
                "harmful_refusal_delta": (
                    float(harmful_refusal_delta)
                    if harmful_refusal_delta is not None
                    else None
                ),
                "harmless_refusal_delta": (
                    float(harmless_refusal_delta)
                    if harmless_refusal_delta is not None
                    else None
                ),
                "score": float(score),
            }
            results.append(row)

            if best is None or row["score"] > best["score"]:
                best = row
                best_direction = direction.detach().clone()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    return best, best_direction, results
