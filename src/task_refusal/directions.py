from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from task_refusal.data import Context
from task_refusal.hooks import temporary_hooks
from task_refusal.progress import ProgressTracker


@torch.inference_mode()
def mean_residual_activations(
    model_harness,
    contexts: Sequence[Context],
    positions: Sequence[int],
    *,
    batch_size: int | None = None,
    progress_stage: str = "mean_residual_activations",
) -> Tensor:
    if not contexts:
        raise ValueError("Cannot compute activations for an empty context list.")
    effective_batch_size = batch_size or model_harness.config.batch_size
    cache = torch.zeros(
        (len(positions), model_harness.n_layers, model_harness.hidden_size),
        dtype=torch.float64,
        device="cpu",
    )

    def make_hook(layer: int):
        def hook(_module, inputs):
            activation = inputs[0] if isinstance(inputs, tuple) else inputs
            if activation.shape[1] < abs(min(positions)):
                raise ValueError(
                    f"Prompt has {activation.shape[1]} tokens but position {min(positions)} "
                    "was requested."
                )
            selected = activation[:, list(positions), :]
            cache[:, layer] += selected.detach().to(torch.float64).cpu().sum(dim=0)

        return hook

    hooks = [(block, make_hook(layer)) for layer, block in enumerate(model_harness.blocks)]
    progress = ProgressTracker(
        progress_stage,
        (len(contexts) + effective_batch_size - 1) // effective_batch_size,
        {"states": len(contexts), "positions": list(positions)},
    )
    for start in range(0, len(contexts), effective_batch_size):
        batch = model_harness.tokenize(contexts[start : start + effective_batch_size])
        with temporary_hooks(pre_hooks=hooks):
            model_harness.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
            )
        progress.advance(
            item=f"states={start}:{min(start + effective_batch_size, len(contexts))}"
        )
    return cache / len(contexts)


def extract_candidate_directions(
    model_harness,
    harmful_contexts: Sequence[Context],
    benign_contexts: Sequence[Context],
    positions: Sequence[int],
) -> Tensor:
    harmful_mean = mean_residual_activations(
        model_harness, harmful_contexts, positions
    )
    benign_mean = mean_residual_activations(model_harness, benign_contexts, positions)
    directions = harmful_mean - benign_mean
    if not torch.isfinite(directions).all():
        raise ValueError("Candidate directions contain NaN or infinity.")
    return directions


def save_candidates(path: str | Path, candidates: Tensor, positions: Sequence[int]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "directions": candidates.to(torch.float32),
            "positions": list(positions),
        },
        output,
    )


def load_candidates(path: str | Path) -> tuple[Tensor, tuple[int, ...]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return payload["directions"], tuple(int(x) for x in payload["positions"])
