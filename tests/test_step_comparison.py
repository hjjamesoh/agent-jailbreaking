from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from task_refusal.config import ModelConfig
from task_refusal.step_comparison import (
    StepComparisonPipeline,
    StepDataConfig,
    StepExperimentConfig,
    StepMethodConfig,
    balanced_indices,
    cohen_d,
    difference_of_means,
    load_instruction_split,
    load_step_config,
    projection_metrics,
    roc_auc,
)


def test_load_instruction_split_builds_user_contexts(tmp_path: Path) -> None:
    path = tmp_path / "harmful_train.json"
    path.write_text(
        json.dumps([{"instruction": "first"}, {"instruction": "second"}]),
        encoding="utf-8",
    )

    contexts = load_instruction_split(path)

    assert contexts == [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
    ]


def test_balanced_indices_is_deterministic_and_balanced() -> None:
    labels = [0, 0, 0, 1, 1]
    positive, negative = balanced_indices(labels, max_per_side=2, min_per_side=2, seed=42)
    repeated = balanced_indices(labels, max_per_side=2, min_per_side=2, seed=42)

    assert (positive, negative) == repeated
    assert len(positive) == len(negative) == 2
    assert all(labels[index] == 1 for index in positive)
    assert all(labels[index] == 0 for index in negative)


def test_balanced_indices_rejects_underpowered_group() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        balanced_indices([0, 0, 1], max_per_side=2, min_per_side=2, seed=42)


def test_direction_and_projection_metrics() -> None:
    activations = torch.tensor(
        [
            [[2.0, 0.0]],
            [[3.0, 0.0]],
            [[-2.0, 0.0]],
            [[-3.0, 0.0]],
        ]
    )
    direction = difference_of_means(activations, [0, 1], [2, 3])
    metrics = projection_metrics(direction[0], activations[:, 0, :], [1, 1, 0, 0])

    assert direction.tolist() == [[5.0, 0.0]]
    assert metrics["auroc"] == 1.0
    assert metrics["mean_difference"] == 5.0
    assert metrics["cohen_d"] > 0


def test_metric_helpers_handle_ties() -> None:
    assert roc_auc([1, 0], [1.0, 1.0]) == 0.5
    assert cohen_d([1, 1, 0, 0], [2.0, 3.0, -2.0, -3.0]) > 0


def test_h2_config_uses_fixed_paper_coordinate() -> None:
    config = load_step_config(Path("configs/h2_agent_step_llama31.yaml"))
    assert config.method.primary_layer == 12
    assert config.method.position == -5
    assert config.model.name == "meta-llama/Llama-3.1-8B-Instruct"
    assert config.data.split_ratios == (0.7, 0.0, 0.3)
    assert config.data.min_test_per_side == 6


def test_analyze_writes_complete_h2_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True)
    config = StepExperimentConfig(
        experiment_name="synthetic-h2",
        output_dir=output_dir,
        model=ModelConfig(
            name="unused",
            revision="unused",
            torch_dtype="float32",
            device_map="cpu",
            max_length=32,
            batch_size=2,
            attn_implementation="eager",
        ),
        data=StepDataConfig(
            agentlens_train=tmp_path / "train.json",
            agentlens_test=tmp_path / "test.json",
            llm_split_dir=tmp_path / "splits",
            split_ratios=(0.7, 0.15, 0.15),
            seed=42,
            max_train_per_side=2,
            max_test_per_side=2,
            min_train_per_side=2,
            min_test_per_side=2,
        ),
        method=StepMethodConfig(
            position=-5,
            primary_layer=1,
            cache_dtype="float16",
            bootstrap_repetitions=5,
        ),
    )
    positive = torch.tensor(
        [[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]
    )
    negative = torch.tensor(
        [
            [[-2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]],
            [[-3.0, 0.0, 0.0], [-4.0, 0.0, 0.0]],
        ]
    )

    def save_cache(name: str, activations: torch.Tensor) -> None:
        torch.save({"activations": activations, "position": -5}, cache_dir / f"{name}.pt")

    for split in ("train", "test"):
        save_cache(f"llm_{split}_harmful", positive)
        save_cache(f"llm_{split}_harmless", negative)
        save_cache(f"agent_{split}", torch.cat([positive, negative]))
        (cache_dir / f"agent_{split}_metadata.json").write_text(
            json.dumps(
                [
                    {"state_id": "p1", "step": 1, "label": 1},
                    {"state_id": "p2", "step": 1, "label": 1},
                    {"state_id": "n1", "step": 1, "label": 0},
                    {"state_id": "n2", "step": 1, "label": 0},
                ]
            ),
            encoding="utf-8",
        )
    (output_dir / "data_summary.json").write_text(
        json.dumps({"agentlens": {"eligible_steps": [1]}}), encoding="utf-8"
    )
    (output_dir / "position_label.json").write_text(json.dumps({"-5": "<|eot_id|>"}))

    StepComparisonPipeline(config).analyze()

    summary = json.loads((output_dir / "analysis" / "summary.json").read_text())
    assert summary["eligible_steps"] == [1]
    assert summary["comparisons"]["agent_step_1"]["primary_cosine_with_llm"] == 1.0
    for filename in (
        "layerwise_cosine.csv",
        "primary_cosine.csv",
        "projection_metrics.csv",
        "bootstrap_cosine.csv",
    ):
        assert (output_dir / "analysis" / filename).stat().st_size > 0
