from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from task_refusal.behavior import WildGuardJudge, evaluate_selected_agent_behavior
from task_refusal.config import ExperimentConfig
from task_refusal.data import AgentState, deterministic_sample, load_states
from task_refusal.directions import (
    load_candidates,
    mean_residual_activations,
    save_candidates,
)
from task_refusal.evaluation import (
    cosine_similarity_matrix,
    cross_task_delta_matrix,
    evaluate_cross_task,
    write_matrix_csv,
)
from task_refusal.hooks import activation_addition_pre_hook
from task_refusal.progress import ProgressTracker, gpu_memory, log_event
from task_refusal.selection import load_selected_direction, select_direction


def _stable_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "little")


def _stratified_category_sample(
    states: Sequence[AgentState],
    count: int,
    categories: Sequence[str],
    seed: int,
) -> list[AgentState]:
    if len(states) < count:
        raise ValueError(f"Requested {count} states but only {len(states)} are eligible.")
    buckets: dict[str, list[AgentState]] = {}
    for category in categories:
        bucket = [state for state in states if state.category == category]
        random.Random(_stable_seed(seed, category)).shuffle(bucket)
        buckets[category] = bucket
    selected: list[AgentState] = []
    offset = 0
    while len(selected) < count:
        added = False
        for category in categories:
            bucket = buckets[category]
            if offset < len(bucket):
                selected.append(bucket[offset])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        offset += 1
    if len(selected) != count:
        raise ValueError(
            f"Could not draw {count} category-stratified states; selected {len(selected)}."
        )
    return selected


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ExperimentPipeline:
    def __init__(self, config: ExperimentConfig, model_harness):
        self.config = config
        self.model_harness = model_harness
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir = config.data.manifest_dir
        mas_manifest = manifest_dir / "agentlens_states.jsonl"
        hazard_manifest = manifest_dir / "agenthazard_states.jsonl"
        self.mas = load_states(mas_manifest)
        self.hazard = load_states(hazard_manifest)
        self.states_by_id = {state.state_id: state for state in (*self.mas, *self.hazard)}
        if len(self.states_by_id) != len(self.mas) + len(self.hazard):
            raise ValueError("Manifest state_id values must be globally unique.")
        self.direction_names = ("global", *config.data.categories)
        self.refusal_token_ids = model_harness.refusal_token_ids(
            config.method.refusal_token_strings
        )
        resolved_path = self.output_dir / "resolved_config.json"
        resolved_payload = json.loads(
            json.dumps(
                {
                    "experiment_name": config.experiment_name,
                    "output_dir": str(config.output_dir),
                    "model": asdict(config.model),
                    "data": asdict(config.data),
                    "method": asdict(config.method),
                    "selection": asdict(config.selection),
                    "evaluation": asdict(config.evaluation),
                    "manifest_sha256": {
                        "agentlens": _file_sha256(mas_manifest),
                        "agenthazard": _file_sha256(hazard_manifest),
                    },
                },
                default=str,
            )
        )
        if resolved_path.exists():
            previous = json.loads(resolved_path.read_text(encoding="utf-8"))
            if previous != resolved_payload:
                raise RuntimeError(
                    "Config or prepared manifests changed for an existing run. "
                    "Use a new output_dir instead of mixing checkpoints."
                )
        else:
            _write_json(resolved_path, resolved_payload)
        log_event(
            "experiment_initialized",
            experiment=config.experiment_name,
            model=config.model.name,
            mas_states=len(self.mas),
            agenthazard_states=len(self.hazard),
            refusal_token_ids=self.refusal_token_ids.tolist(),
            **gpu_memory(),
        )

    def _score_states(
        self, states: Sequence[AgentState], path: Path, name: str
    ) -> dict[str, float]:
        existing_rows = _read_jsonl(path)
        scores = {str(row["state_id"]): float(row["score"]) for row in existing_rows}
        pending = [state for state in states if state.state_id not in scores]
        chunk_size = max(self.config.model.batch_size * 4, 16)
        total = (len(pending) + chunk_size - 1) // chunk_size
        progress = ProgressTracker(f"baseline_score_{name}", total)
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start : start + chunk_size]
            values = self.model_harness.refusal_scores(
                [state.context() for state in chunk], self.refusal_token_ids
            ).tolist()
            rows = []
            for state, value in zip(chunk, values):
                scores[state.state_id] = float(value)
                rows.append({"state_id": state.state_id, "score": float(value)})
            _write_jsonl(path, rows, mode="a")
            progress.advance(
                item=f"states={start}:{start + len(chunk)}", checkpoint=str(path)
            )
        return scores

    def filter_and_split(self, force: bool = False) -> Path:
        split_path = self.output_dir / "data" / "direction_splits.jsonl"
        if split_path.exists() and not force:
            log_event("stage_skipped", stage="filter", reason="checkpoint_exists")
            return split_path
        if force:
            downstream = self.output_dir / "directions"
            if downstream.exists():
                raise RuntimeError(
                    "Refusing to recompute splits while direction checkpoints exist. "
                    "Use a new output_dir or archive the existing run first."
                )
            for score_path in (
                self.output_dir / "data" / "mas_baseline_scores.jsonl",
                self.output_dir / "data" / "hazard_baseline_scores.jsonl",
            ):
                if score_path.exists():
                    score_path.unlink()
        mas_safe = [state for state in self.mas if state.harm_label == 0]
        mas_scores = self._score_states(
            mas_safe, self.output_dir / "data" / "mas_baseline_scores.jsonl", "mas_safe"
        )
        hazard_scores = self._score_states(
            self.hazard,
            self.output_dir / "data" / "hazard_baseline_scores.jsonl",
            "agenthazard",
        )
        safe_eligible = [state for state in mas_safe if mas_scores[state.state_id] < 0]
        rows: list[dict[str, Any]] = []
        summary: dict[str, Any] = {"safe": {}}
        for use_split, count in (
            ("train", self.config.data.direction_train_per_side),
            (
                "val",
                max(
                    self.config.data.selection_val_per_side,
                    self.config.selection.behavior_val_per_side,
                ),
            ),
        ):
            candidates = [state for state in safe_eligible if state.split == use_split]
            selected = deterministic_sample(
                candidates, count, _stable_seed(self.config.data.seed, f"safe-{use_split}")
            )
            rows.extend(
                {
                    "direction_group": "safe",
                    "role": "safe",
                    "use_split": use_split,
                    "state_id": state.state_id,
                    "baseline_refusal_score": mas_scores[state.state_id],
                }
                for state in selected
            )
            summary["safe"][use_split] = {
                "eligible": len(candidates),
                "selected": len(selected),
            }
        safe_test_candidates = [state for state in safe_eligible if state.split == "test"]
        safe_test_count = min(self.config.data.max_test_per_category, len(safe_test_candidates))
        safe_test = deterministic_sample(
            safe_test_candidates,
            safe_test_count,
            _stable_seed(self.config.data.seed, "safe-test"),
        )
        rows.extend(
            {
                "direction_group": "safe",
                "role": "safe",
                "use_split": "test",
                "state_id": state.state_id,
                "baseline_refusal_score": mas_scores[state.state_id],
            }
            for state in safe_test
        )
        summary["safe"]["test"] = {
            "eligible": len(safe_test_candidates),
            "selected": len(safe_test),
        }

        for category in self.config.data.categories:
            summary[category] = {}
            category_states = [state for state in self.hazard if state.category == category]
            for use_split, count in (
                ("train", self.config.data.direction_train_per_side),
                ("val", self.config.data.selection_val_per_side),
            ):
                candidates = [
                    state
                    for state in category_states
                    if state.split == use_split and hazard_scores[state.state_id] > 0
                ]
                selected = deterministic_sample(
                    candidates,
                    count,
                    _stable_seed(self.config.data.seed, f"{category}-{use_split}"),
                )
                rows.extend(
                    {
                        "direction_group": category,
                        "role": "harmful",
                        "use_split": use_split,
                        "state_id": state.state_id,
                        "baseline_refusal_score": hazard_scores[state.state_id],
                    }
                    for state in selected
                )
                summary[category][use_split] = {
                    "eligible_refusal_positive": len(candidates),
                    "selected": len(selected),
                }
            behavior_candidates = [
                state for state in category_states if state.split == "val"
            ]
            behavior_selected = deterministic_sample(
                behavior_candidates,
                self.config.selection.behavior_val_per_side,
                _stable_seed(self.config.data.seed, f"{category}-behavior-val"),
            )
            rows.extend(
                {
                    "direction_group": category,
                    "role": "behavior_harmful",
                    "use_split": "val",
                    "state_id": state.state_id,
                    "baseline_refusal_score": hazard_scores[state.state_id],
                }
                for state in behavior_selected
            )
            summary[category]["behavior_val"] = {
                "eligible": len(behavior_candidates),
                "selected": len(behavior_selected),
            }
            test_candidates = [state for state in category_states if state.split == "test"]
            test_count = min(self.config.data.max_test_per_category, len(test_candidates))
            test = deterministic_sample(
                test_candidates,
                test_count,
                _stable_seed(self.config.data.seed, f"{category}-test"),
            )
            rows.extend(
                {
                    "direction_group": category,
                    "role": "harmful",
                    "use_split": "test",
                    "state_id": state.state_id,
                    "baseline_refusal_score": hazard_scores[state.state_id],
                }
                for state in test
            )
            summary[category]["test"] = {
                "eligible": len(test_candidates),
                "selected": len(test),
            }

        for use_split, count in (
            ("train", self.config.data.direction_train_per_side),
            ("val", self.config.data.selection_val_per_side),
        ):
            candidates = [
                state
                for state in self.hazard
                if state.split == use_split and hazard_scores[state.state_id] > 0
            ]
            selected = _stratified_category_sample(
                candidates,
                count,
                self.config.data.categories,
                _stable_seed(self.config.data.seed, f"global-{use_split}"),
            )
            rows.extend(
                {
                    "direction_group": "global",
                    "role": "harmful",
                    "use_split": use_split,
                    "state_id": state.state_id,
                    "baseline_refusal_score": hazard_scores[state.state_id],
                }
                for state in selected
            )
        global_behavior_candidates = [
            state for state in self.hazard if state.split == "val"
        ]
        global_behavior_selected = _stratified_category_sample(
            global_behavior_candidates,
            self.config.selection.behavior_val_per_side,
            self.config.data.categories,
            _stable_seed(self.config.data.seed, "global-behavior-val"),
        )
        rows.extend(
            {
                "direction_group": "global",
                "role": "behavior_harmful",
                "use_split": "val",
                "state_id": state.state_id,
                "baseline_refusal_score": hazard_scores[state.state_id],
            }
            for state in global_behavior_selected
        )
        summary["global"] = {
            "behavior_val": {
                "eligible": len(global_behavior_candidates),
                "selected": len(global_behavior_selected),
            }
        }
        _write_jsonl(split_path, rows)
        _write_json(self.output_dir / "data" / "filter_summary.json", summary)
        log_event("stage_complete", stage="filter", checkpoint=str(split_path))
        return split_path

    def _direction_states(self) -> dict[tuple[str, str, str], list[AgentState]]:
        path = self.output_dir / "data" / "direction_splits.jsonl"
        if not path.exists():
            raise FileNotFoundError("Run the filter stage before direction extraction.")
        result: dict[tuple[str, str, str], list[AgentState]] = {}
        for row in _read_jsonl(path):
            key = (str(row["direction_group"]), str(row["role"]), str(row["use_split"]))
            result.setdefault(key, []).append(self.states_by_id[str(row["state_id"])])
        return result

    def _selected_names(self, only: Sequence[str] | None) -> tuple[str, ...]:
        if not only:
            return self.direction_names
        invalid = sorted(set(only) - set(self.direction_names))
        if invalid:
            raise ValueError(f"Unknown direction names: {invalid}")
        return tuple(name for name in self.direction_names if name in set(only))

    def extract(self, only: Sequence[str] | None = None) -> None:
        groups = self._direction_states()
        safe = groups[("safe", "safe", "train")]
        labels = self.model_harness.suffix_token_labels(
            safe[0].context(), self.config.method.positions
        )
        _write_json(
            self.output_dir / "directions" / "position_labels.json",
            dict(zip(map(str, self.config.method.positions), labels)),
        )
        safe_mean_path = self.output_dir / "directions" / "safe_mean.pt"
        if safe_mean_path.exists():
            safe_mean = torch.load(safe_mean_path, map_location="cpu", weights_only=True)
            log_event("safe_mean_resumed", checkpoint=str(safe_mean_path))
        else:
            safe_mean = mean_residual_activations(
                self.model_harness,
                [state.context() for state in safe],
                self.config.method.positions,
                progress_stage="direction_mean_safe",
            )
            safe_mean_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(safe_mean, safe_mean_path)
            log_event("safe_mean_complete", checkpoint=str(safe_mean_path))
        for name in self._selected_names(only):
            output = self.output_dir / "directions" / name / "candidates.pt"
            if output.exists():
                log_event("stage_item_skipped", stage="extract", category=name)
                continue
            harmful = groups[(name, "harmful", "train")]
            log_event(
                "direction_extraction_start",
                category=name,
                harmful=len(harmful),
                safe=len(safe),
            )
            harmful_mean = mean_residual_activations(
                self.model_harness,
                [state.context() for state in harmful],
                self.config.method.positions,
                progress_stage=f"direction_mean_{name}",
            )
            candidates = harmful_mean - safe_mean
            save_candidates(output, candidates, self.config.method.positions)
            log_event("direction_extraction_complete", category=name, checkpoint=str(output))
        log_event("stage_complete", stage="extract")

    def select(self, only: Sequence[str] | None = None) -> None:
        groups = self._direction_states()
        safe = groups[("safe", "safe", "val")]
        names = self._selected_names(only)
        pending = [
            name
            for name in names
            if not (self.output_dir / "directions" / name / "selected_direction.pt").exists()
        ]
        for name in names:
            if name not in pending:
                log_event("stage_item_skipped", stage="select", category=name)
        if pending:
            judge = WildGuardJudge(self.config.selection)
            try:
                for name in pending:
                    category_dir = self.output_dir / "directions" / name
                    candidates, positions = load_candidates(category_dir / "candidates.pt")
                    select_direction(
                        self.model_harness,
                        groups[(name, "harmful", "val")],
                        safe,
                        groups[(name, "behavior_harmful", "val")],
                        candidates,
                        positions,
                        self.refusal_token_ids,
                        category_dir,
                        judge=judge,
                        selection_config=self.config.selection,
                        category=name,
                        kl_threshold=self.config.method.kl_threshold,
                        induce_refusal_threshold=self.config.method.induce_refusal_threshold,
                        prune_layer_percentage=self.config.method.prune_layer_percentage,
                        addition_coefficient=self.config.method.addition_coefficient,
                    )
            finally:
                judge.close()
        log_event("stage_complete", stage="select")

    def _load_selected(self) -> dict[str, dict[str, Any]]:
        selected = {}
        for name in self.direction_names:
            path = self.output_dir / "directions" / name / "selected_direction.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing selected direction: {path}")
            selected[name] = load_selected_direction(path)
        return selected

    def _random_controls(
        self,
        selected: dict[str, dict[str, Any]],
        targets: dict[str, Sequence[list[dict[str, str]]]],
        analysis_dir: Path,
    ) -> None:
        rows = []
        generator = torch.Generator(device="cpu").manual_seed(self.config.data.seed)
        progress = ProgressTracker("random_direction_controls", len(self.config.data.categories))
        for category in self.config.data.categories:
            payload = selected[category]
            direction = payload["direction"]
            random_direction = torch.randn(direction.shape, generator=generator)
            random_direction = random_direction / random_direction.norm() * direction.norm()
            hooks = [
                (
                    self.model_harness.blocks[int(payload["layer"])],
                    activation_addition_pre_hook(
                        random_direction,
                        float(
                            payload.get(
                                "coefficient", self.config.method.addition_coefficient
                            )
                        ),
                    ),
                )
            ]
            contexts = targets[category]
            baseline = self.model_harness.refusal_scores(
                contexts,
                self.refusal_token_ids,
                progress_label=f"random_control_{category}_baseline",
            )
            scores = self.model_harness.refusal_scores(
                contexts,
                self.refusal_token_ids,
                pre_hooks=hooks,
                progress_label=f"random_control_{category}_addition",
            )
            rows.append(
                {
                    "category": category,
                    "n": len(contexts),
                    "mean_delta": float((scores - baseline).mean()),
                    "positive_rate": float((scores > 0).float().mean()),
                }
            )
            progress.advance(item=category)
        _write_jsonl(analysis_dir / "random_direction_controls.jsonl", rows)

    def evaluate(self) -> None:
        groups = self._direction_states()
        selected = self._load_selected()
        analysis_dir = self.output_dir / "analysis"
        geometry = cosine_similarity_matrix(
            {name: payload["direction"] for name, payload in selected.items()}
        )
        _write_json(analysis_dir / "cosine_similarity.json", geometry)
        write_matrix_csv(analysis_dir / "cosine_similarity.csv", geometry)
        targets = {
            category: [
                state.context()
                for state in groups[(category, "harmful", "test")]
            ]
            for category in self.config.data.categories
        }
        targets["safe"] = [
            state.context() for state in groups[("safe", "safe", "test")]
        ]
        rows = evaluate_cross_task(
            self.model_harness,
            selected,
            targets,
            self.refusal_token_ids,
            analysis_dir / "cross_task_results.jsonl",
            self.config.method.addition_coefficient,
        )
        for intervention in ("ablation", "addition"):
            matrix = cross_task_delta_matrix(rows, intervention)
            _write_json(analysis_dir / f"{intervention}_delta.json", matrix)
            write_matrix_csv(analysis_dir / f"{intervention}_delta.csv", matrix)
        if self.config.evaluation.include_random_direction:
            self._random_controls(selected, targets, analysis_dir)

        behavior_dir = analysis_dir / "agent_behavior"
        behavior_summary = behavior_dir / "heldout_behavior_summary.jsonl"
        expected_conditions = 3 * len(self.config.data.categories) + 1 + len(selected)
        completed_conditions = {
            (str(row["target"]), str(row["direction"]))
            for row in _read_jsonl(behavior_summary)
        }
        if len(completed_conditions) < expected_conditions:
            harmful_behavior_targets = {
                category: deterministic_sample(
                    groups[(category, "harmful", "test")],
                    self.config.evaluation.behavior_test_per_category,
                    _stable_seed(self.config.data.seed, f"{category}-behavior-test"),
                )
                for category in self.config.data.categories
            }
            safe_behavior_states = deterministic_sample(
                groups[("safe", "safe", "test")],
                self.config.evaluation.behavior_safe_test_size,
                _stable_seed(self.config.data.seed, "safe-behavior-test"),
            )
            judge = WildGuardJudge(self.config.selection)
            try:
                evaluate_selected_agent_behavior(
                    self.model_harness,
                    judge,
                    selected,
                    harmful_behavior_targets,
                    safe_behavior_states,
                    behavior_dir,
                    self.config.selection,
                )
            finally:
                judge.close()
        else:
            log_event(
                "heldout_agent_behavior_skipped",
                reason="checkpoint_complete",
                checkpoint=str(behavior_summary),
            )

        log_event("stage_complete", stage="evaluate", output=str(analysis_dir))
