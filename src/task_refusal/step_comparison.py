from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor

from task_refusal.config import ModelConfig
from task_refusal.data import AgentState, Context, load_agentlens_states, validate_no_group_leakage
from task_refusal.progress import log_event


@dataclass(frozen=True)
class StepDataConfig:
    agentlens_train: Path
    agentlens_test: Path
    llm_split_dir: Path
    split_ratios: tuple[float, float, float]
    seed: int
    max_train_per_side: int
    max_test_per_side: int
    min_train_per_side: int
    min_test_per_side: int


@dataclass(frozen=True)
class StepMethodConfig:
    position: int
    primary_layer: int
    cache_dtype: str
    bootstrap_repetitions: int


@dataclass(frozen=True)
class StepExperimentConfig:
    experiment_name: str
    output_dir: Path
    model: ModelConfig
    data: StepDataConfig
    method: StepMethodConfig


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {key}")
    return mapping[key]


def load_step_config(path: str | Path) -> StepExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("The YAML root must be a mapping.")
    model = _required(raw, "model")
    data = _required(raw, "data")
    method = _required(raw, "method")
    ratios = tuple(float(value) for value in _required(data, "split_ratios"))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("data.split_ratios must contain three values summing to 1.")
    position = int(_required(method, "position"))
    if position >= 0:
        raise ValueError("method.position must be a negative token index.")
    cache_dtype = str(method.get("cache_dtype", "float16"))
    if cache_dtype not in {"float16", "float32"}:
        raise ValueError("method.cache_dtype must be float16 or float32.")
    positive = {
        "max_train_per_side": int(data.get("max_train_per_side", 128)),
        "max_test_per_side": int(data.get("max_test_per_side", 64)),
        "min_train_per_side": int(data.get("min_train_per_side", 16)),
        "min_test_per_side": int(data.get("min_test_per_side", 8)),
        "bootstrap_repetitions": int(method.get("bootstrap_repetitions", 500)),
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError("H2 sample sizes and bootstrap repetitions must be positive.")
    if positive["min_train_per_side"] > positive["max_train_per_side"]:
        raise ValueError("min_train_per_side cannot exceed max_train_per_side.")
    if positive["min_test_per_side"] > positive["max_test_per_side"]:
        raise ValueError("min_test_per_side cannot exceed max_test_per_side.")
    return StepExperimentConfig(
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
        data=StepDataConfig(
            agentlens_train=Path(_required(data, "agentlens_train")),
            agentlens_test=Path(_required(data, "agentlens_test")),
            llm_split_dir=Path(_required(data, "llm_split_dir")),
            split_ratios=ratios,
            seed=int(data.get("seed", 42)),
            max_train_per_side=positive["max_train_per_side"],
            max_test_per_side=positive["max_test_per_side"],
            min_train_per_side=positive["min_train_per_side"],
            min_test_per_side=positive["min_test_per_side"],
        ),
        method=StepMethodConfig(
            position=position,
            primary_layer=int(_required(method, "primary_layer")),
            cache_dtype=cache_dtype,
            bootstrap_repetitions=positive["bootstrap_repetitions"],
        ),
    )


def load_instruction_split(path: Path) -> list[Context]:
    if not path.exists():
        raise FileNotFoundError(
            f"Official refusal-direction split not found: {path}. "
            "Run python -u scripts/prepare_h2_data.py first."
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"Expected a JSON list in {path}")
    contexts: list[Context] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not str(row.get("instruction", "")).strip():
            raise ValueError(f"Invalid instruction row {index} in {path}")
        contexts.append([{"role": "user", "content": str(row["instruction"])}])
    return contexts


def _stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _sample_indices(indices: Sequence[int], count: int, seed: int) -> list[int]:
    values = list(indices)
    random.Random(seed).shuffle(values)
    return sorted(values[:count])


def balanced_indices(
    labels: Sequence[int],
    *,
    max_per_side: int,
    min_per_side: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    harmful = [index for index, label in enumerate(labels) if int(label) == 1]
    benign = [index for index, label in enumerate(labels) if int(label) == 0]
    count = min(max_per_side, len(harmful), len(benign))
    if count < min_per_side:
        raise ValueError(
            f"Need at least {min_per_side} examples per side, found "
            f"harmful={len(harmful)}, benign={len(benign)}."
        )
    return (
        _sample_indices(harmful, count, seed),
        _sample_indices(benign, count, seed + 1),
    )


def difference_of_means(
    activations: Tensor, positive: Sequence[int], negative: Sequence[int]
) -> Tensor:
    if not positive or not negative:
        raise ValueError("Both sides of a direction must be non-empty.")
    values = activations.to(torch.float32)
    direction = values[list(positive)].mean(dim=0) - values[list(negative)].mean(dim=0)
    if not torch.isfinite(direction).all():
        raise ValueError("Direction contains NaN or infinity.")
    return direction


def cosine(left: Tensor, right: Tensor) -> float:
    left64 = left.to(torch.float64).reshape(-1)
    right64 = right.to(torch.float64).reshape(-1)
    left_norm = left64.norm()
    right_norm = right64.norm()
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return float("nan")
    return float(torch.dot(left64, right64) / (left_norm * right_norm))


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if int(label) == 1]
    negatives = [score for label, score in zip(labels, scores) if int(label) == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def cohen_d(labels: Sequence[int], scores: Sequence[float]) -> float:
    positive = [score for label, score in zip(labels, scores) if int(label) == 1]
    negative = [score for label, score in zip(labels, scores) if int(label) == 0]
    if len(positive) < 2 or len(negative) < 2:
        return float("nan")
    mean_positive = sum(positive) / len(positive)
    mean_negative = sum(negative) / len(negative)
    var_positive = sum((value - mean_positive) ** 2 for value in positive) / (len(positive) - 1)
    var_negative = sum((value - mean_negative) ** 2 for value in negative) / (len(negative) - 1)
    pooled = math.sqrt(
        ((len(positive) - 1) * var_positive + (len(negative) - 1) * var_negative)
        / (len(positive) + len(negative) - 2)
    )
    return (mean_positive - mean_negative) / pooled if pooled > 0 else float("nan")


def projection_metrics(
    direction: Tensor, activations: Tensor, labels: Sequence[int]
) -> dict[str, float]:
    unit = direction.to(torch.float32)
    norm = unit.norm()
    if norm <= 1e-12:
        raise ValueError("Cannot project onto a zero-norm direction.")
    scores = (activations.to(torch.float32) @ (unit / norm)).tolist()
    positive = [score for label, score in zip(labels, scores) if int(label) == 1]
    negative = [score for label, score in zip(labels, scores) if int(label) == 0]
    return {
        "n": float(len(scores)),
        "n_harmful": float(len(positive)),
        "n_benign": float(len(negative)),
        "harmful_mean": sum(positive) / len(positive),
        "benign_mean": sum(negative) / len(negative),
        "mean_difference": sum(positive) / len(positive) - sum(negative) / len(negative),
        "auroc": roc_auc(labels, scores),
        "cohen_d": cohen_d(labels, scores),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def bootstrap_direction_cosine(
    left_positive: Tensor,
    left_negative: Tensor,
    right_positive: Tensor,
    right_negative: Tensor,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values: list[float] = []
    tensors = [
        left_positive.to(torch.float32),
        left_negative.to(torch.float32),
        right_positive.to(torch.float32),
        right_negative.to(torch.float32),
    ]
    for _ in range(repetitions):
        means = []
        for tensor in tensors:
            indices = torch.randint(len(tensor), (len(tensor),), generator=generator)
            means.append(tensor[indices].mean(dim=0))
        values.append(cosine(means[0] - means[1], means[2] - means[3]))
    return {
        "mean": sum(values) / len(values),
        "ci95_low": _percentile(values, 0.025),
        "ci95_high": _percentile(values, 0.975),
        "repetitions": float(repetitions),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class StepComparisonPipeline:
    def __init__(self, config: StepExperimentConfig, model_harness=None):
        self.config = config
        self.model_harness = model_harness
        self.output_dir = config.output_dir

    def _agent_states(self) -> list[AgentState]:
        states = load_agentlens_states(
            (self.config.data.agentlens_train, self.config.data.agentlens_test),
            self.config.data.seed,
            self.config.data.split_ratios,
        )
        validate_no_group_leakage(states)
        return states

    def _llm_path(self, harmtype: str, split: str) -> Path:
        return self.config.data.llm_split_dir / f"{harmtype}_{split}.json"

    def prepare(self) -> dict[str, Any]:
        states = self._agent_states()
        llm_counts = {
            f"{harmtype}_{split}": len(load_instruction_split(self._llm_path(harmtype, split)))
            for harmtype in ("harmful", "harmless")
            for split in ("train", "test")
        }
        counts = Counter((state.split, state.step, state.harm_label) for state in states)
        steps = sorted({state.step for state in states})
        eligible_steps = []
        skipped_steps = []
        for step in steps:
            values = {
                f"{split}_label_{label}": counts[(split, step, label)]
                for split in ("train", "test")
                for label in (0, 1)
            }
            eligible = (
                min(values["train_label_0"], values["train_label_1"])
                >= self.config.data.min_train_per_side
                and min(values["test_label_0"], values["test_label_1"])
                >= self.config.data.min_test_per_side
            )
            (eligible_steps if eligible else skipped_steps).append(step)
        summary = {
            "experiment": self.config.experiment_name,
            "agentlens": {
                "states": len(states),
                "groups": len({state.group_id for state in states}),
                "split_counts": dict(Counter(state.split for state in states)),
                "label_counts": {
                    str(key): value
                    for key, value in Counter(state.harm_label for state in states).items()
                },
                "step_counts": [
                    {
                        "split": split,
                        "step": step,
                        "label": label,
                        "count": count,
                    }
                    for (split, step, label), count in sorted(counts.items())
                ],
                "eligible_steps": eligible_steps,
                "skipped_steps": skipped_steps,
            },
            "llm_reference": llm_counts,
            "primary_coordinate": {
                "layer": self.config.method.primary_layer,
                "position": self.config.method.position,
            },
        }
        _write_json(self.output_dir / "data_summary.json", summary)
        _write_json(self.output_dir / "resolved_config.json", asdict(self.config))
        if not eligible_steps:
            log_event(
                "h2_prepare_failed_no_eligible_steps",
                skipped_steps=skipped_steps,
                count_summary=str(self.output_dir / "data_summary.json"),
            )
            raise ValueError(
                "No AgentLens step has enough harmful and benign train/test states. "
                f"Inspect {self.output_dir / 'data_summary.json'} before changing the minimums."
            )
        log_event(
            "h2_prepare_complete",
            eligible_steps=eligible_steps,
            skipped_steps=skipped_steps,
            output=str(self.output_dir / "data_summary.json"),
        )
        return summary

    def _require_harness(self):
        if self.model_harness is None:
            raise RuntimeError("This H2 stage requires the model harness.")

    def _cache_contexts(self, name: str, contexts: Sequence[Context]) -> Path:
        self._require_harness()
        path = self.output_dir / "cache" / f"{name}.pt"
        if path.exists():
            log_event("h2_cache_resumed", name=name, checkpoint=str(path))
            return path
        activations = self.model_harness.resid_pre_at_position(
            contexts,
            self.config.method.position,
            progress_stage=f"h2_cache_{name}",
        )
        dtype = torch.float16 if self.config.method.cache_dtype == "float16" else torch.float32
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "activations": activations.to(dtype),
                "position": self.config.method.position,
                "n_layers": self.model_harness.n_layers,
                "hidden_size": self.model_harness.hidden_size,
            },
            path,
        )
        log_event("h2_cache_complete", name=name, states=len(contexts), checkpoint=str(path))
        return path

    def extract(self) -> None:
        self._require_harness()
        summary_path = self.output_dir / "data_summary.json"
        if not summary_path.exists():
            self.prepare()
        if not 0 <= self.config.method.primary_layer < self.model_harness.n_layers:
            raise ValueError(
                f"primary_layer={self.config.method.primary_layer} is outside "
                f"0..{self.model_harness.n_layers - 1}."
            )
        states = self._agent_states()
        for split in ("train", "test"):
            split_states = [state for state in states if state.split == split]
            cache_path = self._cache_contexts(
                f"agent_{split}", [state.context() for state in split_states]
            )
            metadata_path = self.output_dir / "cache" / f"agent_{split}_metadata.json"
            if not metadata_path.exists():
                _write_json(
                    metadata_path,
                    [
                        {
                            "state_id": state.state_id,
                            "episode_id": state.episode_id,
                            "group_id": state.group_id,
                            "step": state.step,
                            "label": state.harm_label,
                        }
                        for state in split_states
                    ],
                )
            log_event("h2_agent_cache_ready", split=split, checkpoint=str(cache_path))

        for split in ("train", "test"):
            for harmtype in ("harmful", "harmless"):
                contexts = load_instruction_split(self._llm_path(harmtype, split))
                limit = (
                    self.config.data.max_train_per_side
                    if split == "train"
                    else self.config.data.max_test_per_side
                )
                count = min(limit, len(contexts))
                indices = _sample_indices(
                    range(len(contexts)),
                    count,
                    _stable_seed(self.config.data.seed, f"llm-{harmtype}-{split}"),
                )
                self._cache_contexts(
                    f"llm_{split}_{harmtype}", [contexts[index] for index in indices]
                )
        labels = self.model_harness.suffix_token_labels(
            load_instruction_split(self._llm_path("harmless", "train"))[0],
            [self.config.method.position],
        )
        _write_json(
            self.output_dir / "position_label.json",
            {str(self.config.method.position): labels[0]},
        )
        log_event("h2_extract_complete", output=str(self.output_dir / "cache"))

    def _load_cache(self, name: str) -> Tensor:
        path = self.output_dir / "cache" / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing activation cache: {path}. Run extract first.")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if int(payload["position"]) != self.config.method.position:
            raise ValueError(f"Cached position does not match config: {path}")
        return payload["activations"].to(torch.float32)

    def _agent_groups(
        self,
        activations: Tensor,
        metadata: Sequence[dict[str, Any]],
        split: str,
        steps: Sequence[int],
    ) -> tuple[dict[str, tuple[Tensor, Tensor]], dict[str, tuple[Tensor, list[int]]]]:
        train_or_test_limit = (
            self.config.data.max_train_per_side
            if split == "train"
            else self.config.data.max_test_per_side
        )
        minimum = (
            self.config.data.min_train_per_side
            if split == "train"
            else self.config.data.min_test_per_side
        )
        pairs: dict[str, tuple[Tensor, Tensor]] = {}
        targets: dict[str, tuple[Tensor, list[int]]] = {}
        all_positive: list[Tensor] = []
        all_negative: list[Tensor] = []
        for step in steps:
            matching = [index for index, row in enumerate(metadata) if int(row["step"]) == step]
            labels = [int(metadata[index]["label"]) for index in matching]
            positive_local, negative_local = balanced_indices(
                labels,
                max_per_side=train_or_test_limit,
                min_per_side=minimum,
                seed=_stable_seed(self.config.data.seed, f"agent-{split}-step-{step}"),
            )
            positive = [matching[index] for index in positive_local]
            negative = [matching[index] for index in negative_local]
            name = f"agent_step_{step}"
            positive_tensor = activations[positive]
            negative_tensor = activations[negative]
            pairs[name] = (positive_tensor, negative_tensor)
            target = torch.cat([positive_tensor, negative_tensor], dim=0)
            targets[name] = (target, [1] * len(positive) + [0] * len(negative))
            all_positive.append(positive_tensor)
            all_negative.append(negative_tensor)
        # Each step contributes its own balanced set. Directions computed from
        # these concatenations therefore avoid long or common steps dominating.
        pooled_positive = torch.cat(all_positive, dim=0)
        pooled_negative = torch.cat(all_negative, dim=0)
        pairs["agent_all"] = (pooled_positive, pooled_negative)
        targets["agent_all"] = (
            torch.cat([pooled_positive, pooled_negative], dim=0),
            [1] * len(pooled_positive) + [0] * len(pooled_negative),
        )
        return pairs, targets

    def analyze(self) -> None:
        summary_path = self.output_dir / "data_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError("Run the H2 prepare stage before analyze.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        steps = [int(step) for step in summary["agentlens"]["eligible_steps"]]
        primary_layer = self.config.method.primary_layer

        llm_train_positive = self._load_cache("llm_train_harmful")
        llm_train_negative = self._load_cache("llm_train_harmless")
        llm_test_positive = self._load_cache("llm_test_harmful")
        llm_test_negative = self._load_cache("llm_test_harmless")
        directions: dict[str, Tensor] = {
            "llm_reference": llm_train_positive.mean(dim=0) - llm_train_negative.mean(dim=0)
        }
        train_pairs: dict[str, tuple[Tensor, Tensor]] = {
            "llm_reference": (llm_train_positive, llm_train_negative)
        }
        test_targets: dict[str, tuple[Tensor, list[int]]] = {
            "llm_reference": (
                torch.cat([llm_test_positive, llm_test_negative], dim=0),
                [1] * len(llm_test_positive) + [0] * len(llm_test_negative),
            )
        }

        for split in ("train", "test"):
            activations = self._load_cache(f"agent_{split}")
            metadata = json.loads(
                (self.output_dir / "cache" / f"agent_{split}_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            if len(activations) != len(metadata):
                raise ValueError(f"Agent {split} cache and metadata length differ.")
            pairs, targets = self._agent_groups(activations, metadata, split, steps)
            if split == "train":
                train_pairs.update(pairs)
                for name, (positive, negative) in pairs.items():
                    directions[name] = positive.mean(dim=0) - negative.mean(dim=0)
            else:
                test_targets.update(targets)

        directions_path = self.output_dir / "directions" / "step_directions.pt"
        directions_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "directions": {name: value.to(torch.float32) for name, value in directions.items()},
                "position": self.config.method.position,
                "primary_layer": primary_layer,
                "steps": steps,
            },
            directions_path,
        )

        names = list(directions)
        layerwise_rows: list[dict[str, Any]] = []
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                for layer in range(directions[left_name].shape[0]):
                    layerwise_rows.append(
                        {
                            "source": left_name,
                            "target": right_name,
                            "layer": layer,
                            "cosine": cosine(
                                directions[left_name][layer], directions[right_name][layer]
                            ),
                            "source_norm": float(directions[left_name][layer].norm()),
                            "target_norm": float(directions[right_name][layer].norm()),
                        }
                    )
        analysis_dir = self.output_dir / "analysis"
        _write_csv(analysis_dir / "layerwise_cosine.csv", layerwise_rows)

        primary_rows: list[dict[str, Any]] = []
        for source_name in names:
            for target_name in names:
                primary_rows.append(
                    {
                        "source": source_name,
                        "target": target_name,
                        "layer": primary_layer,
                        "position": self.config.method.position,
                        "cosine": cosine(
                            directions[source_name][primary_layer],
                            directions[target_name][primary_layer],
                        ),
                    }
                )
        _write_csv(analysis_dir / "primary_cosine.csv", primary_rows)

        metric_rows: list[dict[str, Any]] = []
        for source_name, direction in directions.items():
            for target_name, (target_activations, target_labels) in test_targets.items():
                metrics = projection_metrics(
                    direction[primary_layer],
                    target_activations[:, primary_layer, :],
                    target_labels,
                )
                metric_rows.append(
                    {
                        "source_direction": source_name,
                        "target_dataset": target_name,
                        "layer": primary_layer,
                        "position": self.config.method.position,
                        **metrics,
                    }
                )
        _write_csv(analysis_dir / "projection_metrics.csv", metric_rows)

        bootstrap_rows = []
        llm_positive = train_pairs["llm_reference"][0][:, primary_layer, :]
        llm_negative = train_pairs["llm_reference"][1][:, primary_layer, :]
        for name in names:
            if name == "llm_reference":
                continue
            positive, negative = train_pairs[name]
            metrics = bootstrap_direction_cosine(
                llm_positive,
                llm_negative,
                positive[:, primary_layer, :],
                negative[:, primary_layer, :],
                self.config.method.bootstrap_repetitions,
                _stable_seed(self.config.data.seed, f"bootstrap-{name}"),
            )
            bootstrap_rows.append(
                {
                    "source": "llm_reference",
                    "target": name,
                    "layer": primary_layer,
                    "position": self.config.method.position,
                    **metrics,
                }
            )
        _write_csv(analysis_dir / "bootstrap_cosine.csv", bootstrap_rows)

        comparisons = {}
        for name in names:
            if name == "llm_reference":
                continue
            values = [
                cosine(directions["llm_reference"][layer], directions[name][layer])
                for layer in range(directions[name].shape[0])
            ]
            finite = [(layer, value) for layer, value in enumerate(values) if math.isfinite(value)]
            best_layer, best_value = max(finite, key=lambda item: item[1])
            matched_metric = next(
                row
                for row in metric_rows
                if row["source_direction"] == name and row["target_dataset"] == name
            )
            llm_transfer_metric = next(
                row
                for row in metric_rows
                if row["source_direction"] == "llm_reference" and row["target_dataset"] == name
            )
            comparisons[name] = {
                "primary_cosine_with_llm": values[primary_layer],
                "best_same_layer_cosine_with_llm": best_value,
                "best_same_layer": best_layer,
                "own_direction_test_auroc": matched_metric["auroc"],
                "llm_direction_test_auroc": llm_transfer_metric["auroc"],
                "own_direction_test_cohen_d": matched_metric["cohen_d"],
                "llm_direction_test_cohen_d": llm_transfer_metric["cohen_d"],
            }
        final_summary = {
            "experiment": self.config.experiment_name,
            "scope": "representation-only H2; no generation, steering, probes, or category routing",
            "position": self.config.method.position,
            "position_label": json.loads(
                (self.output_dir / "position_label.json").read_text(encoding="utf-8")
            )[str(self.config.method.position)],
            "primary_layer": primary_layer,
            "eligible_steps": steps,
            "comparisons": comparisons,
            "artifacts": {
                "directions": str(directions_path),
                "layerwise_cosine": str(analysis_dir / "layerwise_cosine.csv"),
                "primary_cosine": str(analysis_dir / "primary_cosine.csv"),
                "projection_metrics": str(analysis_dir / "projection_metrics.csv"),
                "bootstrap_cosine": str(analysis_dir / "bootstrap_cosine.csv"),
            },
        }
        _write_json(analysis_dir / "summary.json", final_summary)
        log_event(
            "h2_analysis_complete",
            directions=names,
            primary_layer=primary_layer,
            position=self.config.method.position,
            output=str(analysis_dir / "summary.json"),
        )
