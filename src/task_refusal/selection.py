from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor

from task_refusal.behavior import (
    WildGuardJudge,
    proxy_shortlist,
    select_by_agent_behavior,
)
from task_refusal.config import SelectionConfig
from task_refusal.data import AgentState
from task_refusal.hooks import (
    activation_addition_pre_hook,
    all_direction_ablation_hooks,
)
from task_refusal.modeling import forward_kl_divergence
from task_refusal.progress import ProgressTracker, log_event


def _read_completed(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    completed: dict[tuple[int, int], dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            completed[(int(row["position"]), int(row["layer"]))] = row
    return completed


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def select_direction(
    model_harness,
    harmful_states: Sequence[AgentState],
    benign_states: Sequence[AgentState],
    behavior_harmful_states: Sequence[AgentState],
    candidates: Tensor,
    positions: Sequence[int],
    refusal_token_ids: Tensor,
    output_dir: str | Path,
    *,
    judge: WildGuardJudge,
    selection_config: SelectionConfig,
    category: str,
    kl_threshold: float,
    induce_refusal_threshold: float,
    prune_layer_percentage: float,
    addition_coefficient: float,
) -> dict[str, Any]:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    evaluations_path = artifact_dir / "candidate_evaluations.jsonl"
    existing = _read_completed(evaluations_path)
    harmful_instructions = [state.context() for state in harmful_states]
    benign_instructions = [state.context() for state in benign_states]

    log_event("selection_baseline_start", category=category)
    baseline_harmful = model_harness.refusal_scores(
        harmful_instructions, refusal_token_ids
    )
    baseline_benign = model_harness.refusal_scores(benign_instructions, refusal_token_ids)
    baseline_benign_logits = model_harness.last_logits(benign_instructions)
    log_event(
        "selection_baseline_complete",
        category=category,
        harmful_mean=float(baseline_harmful.mean()),
        benign_mean=float(baseline_benign.mean()),
    )

    total = len(positions) * model_harness.n_layers
    progress = ProgressTracker("candidate_selection", total, {"category": category})
    results = dict(existing)

    for position_index, position in enumerate(positions):
        for source_layer in range(model_harness.n_layers):
            key = (int(position), source_layer)
            if key in results:
                progress.advance(
                    item=f"position={position},layer={source_layer} (resumed)",
                    checkpoint=str(evaluations_path),
                )
                continue

            direction = candidates[position_index, source_layer]
            ablation_pre_hooks, ablation_forward_hooks = all_direction_ablation_hooks(
                model_harness, direction
            )
            harmful_ablation = model_harness.refusal_scores(
                harmful_instructions,
                refusal_token_ids,
                pre_hooks=ablation_pre_hooks,
                forward_hooks=ablation_forward_hooks,
            )
            benign_ablation_logits = model_harness.last_logits(
                benign_instructions,
                pre_hooks=ablation_pre_hooks,
                forward_hooks=ablation_forward_hooks,
            )
            kl_score = forward_kl_divergence(
                baseline_benign_logits, benign_ablation_logits
            ).mean()

            addition_pre_hooks = [
                (
                    model_harness.blocks[source_layer],
                    activation_addition_pre_hook(direction, addition_coefficient),
                )
            ]
            benign_addition = model_harness.refusal_scores(
                benign_instructions,
                refusal_token_ids,
                pre_hooks=addition_pre_hooks,
            )
            row = {
                "category": category,
                "position": int(position),
                "position_index": position_index,
                "layer": source_layer,
                "ablation_refusal_score": float(harmful_ablation.mean()),
                "steering_refusal_score": float(benign_addition.mean()),
                "kl_divergence": float(kl_score),
                "baseline_harmful_score": float(baseline_harmful.mean()),
                "baseline_benign_score": float(baseline_benign.mean()),
            }
            _append_jsonl(evaluations_path, row)
            results[key] = row
            progress.advance(
                item=f"position={position},layer={source_layer}",
                checkpoint=str(evaluations_path),
            )

    last_allowed_layer = int(model_harness.n_layers * (1.0 - prune_layer_percentage))
    eligible = []
    for row in results.values():
        values = (
            row["ablation_refusal_score"],
            row["steering_refusal_score"],
            row["kl_divergence"],
        )
        if any(math.isnan(float(value)) for value in values):
            continue
        if int(row["layer"]) >= last_allowed_layer:
            continue
        if float(row["kl_divergence"]) > kl_threshold:
            continue
        if float(row["steering_refusal_score"]) < induce_refusal_threshold:
            continue
        eligible.append(row)
    if not eligible:
        raise RuntimeError(
            f"All candidate directions for {category} were filtered out. "
            f"Inspect {evaluations_path} before changing thresholds."
        )

    shortlist = proxy_shortlist(eligible, selection_config.shortlist_size)
    (artifact_dir / "proxy_shortlist.json").write_text(
        json.dumps(shortlist, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_event(
        "proxy_shortlist_complete",
        category=category,
        eligible_candidates=len(eligible),
        shortlisted_candidates=len(shortlist),
        checkpoint=str(artifact_dir / "proxy_shortlist.json"),
    )
    selected, selected_direction, behavior_rows = select_by_agent_behavior(
        model_harness,
        judge,
        behavior_harmful_states,
        benign_states,
        candidates,
        shortlist,
        artifact_dir,
        selection_config,
    )
    selected_layer = int(selected["layer"])
    torch.save(
        {
            "direction": selected_direction,
            "category": category,
            "position": int(selected["position"]),
            "layer": selected_layer,
            "coefficient": float(selected["alpha"]),
            "normalization": "unit_l2",
            "selection": selected,
        },
        artifact_dir / "selected_direction.pt",
    )
    metadata = {
        "category": category,
        "selected": selected,
        "eligible_candidates": len(eligible),
        "shortlisted_candidates": len(shortlist),
        "behavior_evaluations": len(behavior_rows),
        "total_candidates": total,
        "selection_method": "full_agent_action_wildguard_with_benign_constraints",
        "thresholds": {
            "kl_threshold": kl_threshold,
            "induce_refusal_threshold": induce_refusal_threshold,
            "prune_layer_percentage": prune_layer_percentage,
            "max_safe_refusal_increase": selection_config.max_safe_refusal_increase,
            "max_safe_nll_increase": selection_config.max_safe_nll_increase,
            "max_judge_parse_error_rate": selection_config.max_judge_parse_error_rate,
            "min_harmful_safety_gain": selection_config.min_harmful_safety_gain,
            "min_harmful_refusal_increase": (
                selection_config.min_harmful_refusal_increase
            ),
        },
    }
    (artifact_dir / "selected_direction.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_event(
        "direction_selected",
        category=category,
        position=selected["position"],
        layer=selected_layer,
        alpha=selected["alpha"],
        harmful_response_harmful_rate=selected["harmful_response_harmful_rate"],
        harmful_refusal_rate=selected["harmful_refusal_rate"],
        safe_refusal_increase=selected["safe_refusal_increase"],
        safe_nll_increase=selected["safe_nll_increase"],
        constraint_fallback=selected["constraint_fallback"],
        ablation_refusal_score=selected["ablation_refusal_score"],
        steering_refusal_score=selected["steering_refusal_score"],
        kl_divergence=selected["kl_divergence"],
    )
    return metadata


def load_selected_direction(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu", weights_only=True)
