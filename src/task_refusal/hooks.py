from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable

import torch
from torch import Tensor, nn


PreHook = tuple[nn.Module, Callable]
ForwardHook = tuple[nn.Module, Callable]


@contextmanager
def temporary_hooks(
    pre_hooks: Iterable[PreHook] = (),
    forward_hooks: Iterable[ForwardHook] = (),
):
    handles = []
    try:
        for module, hook in pre_hooks:
            handles.append(module.register_forward_pre_hook(hook))
        for module, hook in forward_hooks:
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _project_out(activation: Tensor, direction: Tensor) -> Tensor:
    unit = direction.to(device=activation.device, dtype=activation.dtype)
    unit = unit / unit.norm().clamp_min(1e-8)
    return activation - (activation @ unit).unsqueeze(-1) * unit


def direction_ablation_pre_hook(direction: Tensor):
    def hook(_module, inputs):
        activation = inputs[0] if isinstance(inputs, tuple) else inputs
        updated = _project_out(activation, direction)
        if isinstance(inputs, tuple):
            return (updated, *inputs[1:])
        return updated

    return hook


def direction_ablation_output_hook(direction: Tensor):
    def hook(_module, _inputs, output):
        activation = output[0] if isinstance(output, tuple) else output
        updated = _project_out(activation, direction)
        if isinstance(output, tuple):
            return (updated, *output[1:])
        return updated

    return hook


def activation_addition_pre_hook(direction: Tensor, coefficient: float):
    def hook(_module, inputs):
        activation = inputs[0] if isinstance(inputs, tuple) else inputs
        vector = direction.to(device=activation.device, dtype=activation.dtype)
        updated = activation + coefficient * vector
        if isinstance(inputs, tuple):
            return (updated, *inputs[1:])
        return updated

    return hook


def all_direction_ablation_hooks(model_harness, direction: Tensor):
    pre_hooks = [
        (block, direction_ablation_pre_hook(direction)) for block in model_harness.blocks
    ]
    forward_hooks = [
        (attention, direction_ablation_output_hook(direction))
        for attention in model_harness.attention_modules
    ]
    forward_hooks.extend(
        (mlp, direction_ablation_output_hook(direction))
        for mlp in model_harness.mlp_modules
    )
    return pre_hooks, forward_hooks
