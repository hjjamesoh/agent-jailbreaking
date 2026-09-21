from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from task_refusal.hooks import activation_addition_pre_hook, all_direction_ablation_hooks
from task_refusal.progress import ProgressTracker, log_event


def cosine_similarity_matrix(directions: Mapping[str, Tensor]) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for left_name, left in directions.items():
        matrix[left_name] = {}
        for right_name, right in directions.items():
            score = torch.nn.functional.cosine_similarity(
                left.to(torch.float64).reshape(1, -1),
                right.to(torch.float64).reshape(1, -1),
            ).item()
            matrix[left_name][right_name] = float(score)
    return matrix


def write_matrix_csv(path: str | Path, matrix: Mapping[str, Mapping[str, float]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(matrix)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", *columns])
        for row_name in columns:
            writer.writerow([row_name, *(matrix[row_name][column] for column in columns)])


def _load_existing(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[(row["direction"], row["target"], row["intervention"])] = row
    return rows


def evaluate_cross_task(
    model_harness,
    selected: Mapping[str, dict[str, Any]],
    target_prompts: Mapping[str, Sequence[str]],
    refusal_token_ids: Tensor,
    output_path: str | Path,
    addition_coefficient: float,
) -> list[dict[str, Any]]:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_existing(path)

    baseline: dict[str, Tensor] = {}
    for target, prompts in target_prompts.items():
        log_event("evaluation_baseline_start", target=target, examples=len(prompts))
        baseline[target] = model_harness.refusal_scores(
            prompts,
            refusal_token_ids,
            progress_label=f"cross_task_baseline_{target}",
        )
        log_event(
            "evaluation_baseline_complete",
            target=target,
            mean_refusal_score=float(baseline[target].mean()),
        )

    total = len(selected) * len(target_prompts) * 2
    progress = ProgressTracker("cross_task_evaluation", total)
    for direction_name, payload in selected.items():
        direction = payload["direction"]
        source_layer = int(payload["layer"])
        for target, prompts in target_prompts.items():
            for intervention in ("ablation", "addition"):
                key = (direction_name, target, intervention)
                if key in completed:
                    progress.advance(item=f"{direction_name}->{target}:{intervention} (resumed)")
                    continue
                if intervention == "ablation":
                    pre_hooks, forward_hooks = all_direction_ablation_hooks(
                        model_harness, direction
                    )
                else:
                    effective_coefficient = float(
                        payload.get("coefficient", addition_coefficient)
                    )
                    pre_hooks = [
                        (
                            model_harness.blocks[source_layer],
                            activation_addition_pre_hook(direction, effective_coefficient),
                        )
                    ]
                    forward_hooks = []
                scores = model_harness.refusal_scores(
                    prompts,
                    refusal_token_ids,
                    pre_hooks=pre_hooks,
                    forward_hooks=forward_hooks,
                    progress_label=(
                        f"cross_task_{direction_name}_{target}_{intervention}"
                    ),
                )
                row = {
                    "direction": direction_name,
                    "source_layer": source_layer,
                    "source_position": int(payload["position"]),
                    "coefficient": (
                        0.0
                        if intervention == "ablation"
                        else float(payload.get("coefficient", addition_coefficient))
                    ),
                    "target": target,
                    "intervention": intervention,
                    "n": len(prompts),
                    "baseline_mean": float(baseline[target].mean()),
                    "intervention_mean": float(scores.mean()),
                    "mean_delta": float((scores - baseline[target]).mean()),
                    "baseline_positive_rate": float((baseline[target] > 0).float().mean()),
                    "intervention_positive_rate": float((scores > 0).float().mean()),
                }
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                completed[key] = row
                progress.advance(
                    item=f"{direction_name}->{target}:{intervention}",
                    checkpoint=str(path),
                )
    return list(completed.values())


def cross_task_delta_matrix(
    rows: Sequence[dict[str, Any]], intervention: str
) -> dict[str, dict[str, float]]:
    matching = [row for row in rows if row["intervention"] == intervention]
    directions = sorted({row["direction"] for row in matching})
    targets = sorted({row["target"] for row in matching})
    lookup = {(row["direction"], row["target"]): row for row in matching}
    return {
        direction: {
            target: float(lookup[(direction, target)]["mean_delta"]) for target in targets
        }
        for direction in directions
    }
