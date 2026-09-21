from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    name: str
    revision: str
    torch_dtype: str
    device_map: str
    max_length: int
    batch_size: int
    attn_implementation: str


@dataclass(frozen=True)
class DataConfig:
    agentlens_train: Path
    agentlens_test: Path
    agenthazard_dataset: Path
    agenthazard_trace_globs: tuple[str, ...]
    manifest_dir: Path
    seed: int
    split_ratios: tuple[float, float, float]
    direction_train_per_side: int
    selection_val_per_side: int
    max_test_per_category: int
    categories: tuple[str, ...]


@dataclass(frozen=True)
class MethodConfig:
    positions: tuple[int, ...]
    refusal_token_strings: tuple[str, ...]
    kl_threshold: float
    induce_refusal_threshold: float
    prune_layer_percentage: float
    addition_coefficient: float


@dataclass(frozen=True)
class SelectionConfig:
    shortlist_size: int
    behavior_val_per_side: int
    alpha_grid: tuple[float, ...]
    max_new_tokens: int
    generation_batch_size: int
    max_safe_refusal_increase: float
    max_safe_nll_increase: float
    max_judge_parse_error_rate: float
    min_harmful_safety_gain: float
    min_harmful_refusal_increase: float
    judge_model: str
    judge_revision: str
    judge_torch_dtype: str
    judge_device_map: str
    judge_batch_size: int
    judge_max_length: int
    judge_max_new_tokens: int


@dataclass(frozen=True)
class EvaluationConfig:
    include_random_direction: bool
    behavior_test_per_category: int
    behavior_safe_test_size: int


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    output_dir: Path
    model: ModelConfig
    data: DataConfig
    method: MethodConfig
    selection: SelectionConfig
    evaluation: EvaluationConfig


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> ExperimentConfig:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("The YAML root must be a mapping.")
    model = _required(raw, "model")
    data = _required(raw, "data")
    method = _required(raw, "method")
    selection = _required(raw, "selection")
    evaluation = _required(raw, "evaluation")
    ratios = tuple(float(x) for x in _required(data, "split_ratios"))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("data.split_ratios must contain three values summing to 1.")
    alpha_grid = tuple(float(x) for x in selection.get("alpha_grid", [0.5, 1, 2, 4]))
    if not alpha_grid or any(value <= 0 for value in alpha_grid):
        raise ValueError("selection.alpha_grid must contain positive values.")
    if int(selection.get("shortlist_size", 8)) <= 0:
        raise ValueError("selection.shortlist_size must be positive.")
    positive_selection_fields = {
        "behavior_val_per_side": 32,
        "max_new_tokens": 128,
        "generation_batch_size": 4,
        "judge_batch_size": 8,
        "judge_max_length": 4096,
        "judge_max_new_tokens": 32,
    }
    if any(
        int(selection.get(key, default)) <= 0
        for key, default in positive_selection_fields.items()
    ):
        raise ValueError(
            "Selection counts, batch sizes, and token limits must all be positive."
        )
    nonnegative_selection_fields = {
        "max_safe_refusal_increase": 0.05,
        "max_safe_nll_increase": 0.20,
        "max_judge_parse_error_rate": 0.05,
        "min_harmful_safety_gain": 0.0,
        "min_harmful_refusal_increase": 0.0,
    }
    if any(
        float(selection.get(key, default)) < 0
        for key, default in nonnegative_selection_fields.items()
    ):
        raise ValueError("Selection constraint thresholds must be non-negative.")
    if int(evaluation.get("behavior_test_per_category", 64)) <= 0:
        raise ValueError("evaluation.behavior_test_per_category must be positive.")
    if int(evaluation.get("behavior_safe_test_size", 32)) <= 0:
        raise ValueError("evaluation.behavior_safe_test_size must be positive.")
    positions = tuple(int(x) for x in _required(method, "positions"))
    if not positions or len(set(positions)) != len(positions) or any(x >= 0 for x in positions):
        raise ValueError("method.positions must contain unique negative token indices.")
    categories = tuple(str(x) for x in _required(data, "categories"))
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("data.categories must be non-empty and unique.")
    return ExperimentConfig(
        experiment_name=str(_required(raw, "experiment_name")),
        output_dir=Path(_required(raw, "output_dir")),
        model=ModelConfig(
            name=str(_required(model, "name")),
            revision=str(model.get("revision", "main")),
            torch_dtype=str(model.get("torch_dtype", "bfloat16")),
            device_map=str(model.get("device_map", "auto")),
            max_length=int(model.get("max_length", 4096)),
            batch_size=int(model.get("batch_size", 8)),
            attn_implementation=str(model.get("attn_implementation", "eager")),
        ),
        data=DataConfig(
            agentlens_train=Path(_required(data, "agentlens_train")),
            agentlens_test=Path(_required(data, "agentlens_test")),
            agenthazard_dataset=Path(_required(data, "agenthazard_dataset")),
            agenthazard_trace_globs=tuple(_required(data, "agenthazard_trace_globs")),
            manifest_dir=Path(_required(data, "manifest_dir")),
            seed=int(data.get("seed", 42)),
            split_ratios=ratios,
            direction_train_per_side=int(data.get("direction_train_per_side", 128)),
            selection_val_per_side=int(data.get("selection_val_per_side", 32)),
            max_test_per_category=int(data.get("max_test_per_category", 256)),
            categories=categories,
        ),
        method=MethodConfig(
            positions=positions,
            refusal_token_strings=tuple(str(x) for x in _required(method, "refusal_token_strings")),
            kl_threshold=float(method.get("kl_threshold", 0.1)),
            induce_refusal_threshold=float(method.get("induce_refusal_threshold", 0.0)),
            prune_layer_percentage=float(method.get("prune_layer_percentage", 0.2)),
            addition_coefficient=float(method.get("addition_coefficient", 1.0)),
        ),
        selection=SelectionConfig(
            shortlist_size=int(selection.get("shortlist_size", 8)),
            behavior_val_per_side=int(selection.get("behavior_val_per_side", 32)),
            alpha_grid=alpha_grid,
            max_new_tokens=int(selection.get("max_new_tokens", 128)),
            generation_batch_size=int(selection.get("generation_batch_size", 4)),
            max_safe_refusal_increase=float(
                selection.get("max_safe_refusal_increase", 0.05)
            ),
            max_safe_nll_increase=float(selection.get("max_safe_nll_increase", 0.20)),
            max_judge_parse_error_rate=float(
                selection.get("max_judge_parse_error_rate", 0.05)
            ),
            min_harmful_safety_gain=float(
                selection.get("min_harmful_safety_gain", 0.0)
            ),
            min_harmful_refusal_increase=float(
                selection.get("min_harmful_refusal_increase", 0.0)
            ),
            judge_model=str(selection.get("judge_model", "allenai/wildguard")),
            judge_revision=str(selection.get("judge_revision", "main")),
            judge_torch_dtype=str(selection.get("judge_torch_dtype", "bfloat16")),
            judge_device_map=str(selection.get("judge_device_map", "auto")),
            judge_batch_size=int(selection.get("judge_batch_size", 8)),
            judge_max_length=int(selection.get("judge_max_length", 4096)),
            judge_max_new_tokens=int(selection.get("judge_max_new_tokens", 32)),
        ),
        evaluation=EvaluationConfig(
            include_random_direction=bool(evaluation.get("include_random_direction", True)),
            behavior_test_per_category=int(
                evaluation.get("behavior_test_per_category", 64)
            ),
            behavior_safe_test_size=int(evaluation.get("behavior_safe_test_size", 64)),
        ),
    )
