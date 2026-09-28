"""Paired, task-level uncertainty summaries for the held-out test fold."""

from __future__ import annotations

import math
import random
from typing import Mapping


def paired_effect(base: Mapping[str, bool], changed: Mapping[str, bool], *,
                  seed: int = 42, resamples: int = 4000) -> dict[str, float | int]:
    ids = sorted(set(base) & set(changed))
    if not ids:
        raise ValueError("No common judged task IDs")
    if any(type(base[i]) is not bool or type(changed[i]) is not bool for i in ids):
        raise ValueError("Only complete boolean labels can enter a paired effect")
    diffs = [int(changed[i]) - int(base[i]) for i in ids]
    n = len(diffs)
    rng = random.Random(seed)
    boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n
                  for _ in range(resamples))
    return {"n": n, "base_rate": sum(base[i] for i in ids) / n,
            "changed_rate": sum(changed[i] for i in ids) / n,
            "delta": sum(diffs) / n,
            "ci95_low": boot[int(0.025 * (resamples - 1))],
            "ci95_high": boot[int(0.975 * (resamples - 1))]}


def valid_labels(judgments: Mapping[str, dict], key: str) -> dict[str, bool]:
    return {task_id: row[key] for task_id, row in judgments.items()
            if not row.get("parse_error") and not row.get("uncertain")
            and type(row.get(key)) is bool}


def paired_sign_pvalue(first: Mapping[str, bool], second: Mapping[str, bool]) -> float:
    """Exact two-sided paired sign/McNemar p for a binary arm comparison."""
    ids = set(first) & set(second)
    left = sum(first[i] and not second[i] for i in ids)
    right = sum(second[i] and not first[i] for i in ids)
    discordant = left + right
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left, right) + 1))
    return min(1.0, 2 * tail / (2 ** discordant))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    for index, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * p_value))
        adjusted[name] = running
    return adjusted
