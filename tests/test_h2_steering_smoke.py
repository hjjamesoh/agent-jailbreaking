from __future__ import annotations

from pathlib import Path

import pytest
import torch

from task_refusal.h2_steering_smoke import (
    condition_name,
    load_smoke_directions,
    response_features,
    summarize_smoke_rows,
)


def test_response_features_distinguish_refusal_and_tool_call() -> None:
    refusal = response_features("Sorry, but I cannot execute that request.")
    tool = response_features('<tool_call>{"name": "read_file"}</tool_call>')

    assert refusal == {"empty": False, "refusal_phrase": True, "tool_call_like": False}
    assert tool == {"empty": False, "refusal_phrase": False, "tool_call_like": True}
    assert response_features("   ")["empty"] is True


def test_load_smoke_directions_normalizes_selected_layer(tmp_path: Path) -> None:
    path = tmp_path / "step_directions.pt"
    torch.save(
        {
            "directions": {"agent_all": torch.tensor([[3.0, 4.0], [0.0, 2.0]])},
            "position": -5,
            "primary_layer": 1,
            "steps": [1],
        },
        path,
    )

    directions, metadata = load_smoke_directions(path, ["agent_all"], 1)

    assert torch.allclose(directions["agent_all"], torch.tensor([0.0, 1.0]))
    assert metadata["raw_direction_norms"]["agent_all"] == 2.0
    with pytest.raises(ValueError, match="Unknown H2 direction"):
        load_smoke_directions(path, ["missing"], 1)


def test_summary_reports_change_from_label_matched_baseline() -> None:
    rows = [
        {
            "condition": "baseline",
            "label": 1,
            "refusal_phrase": False,
            "tool_call_like": True,
            "empty": False,
        },
        {
            "condition": "candidate",
            "label": 1,
            "refusal_phrase": True,
            "tool_call_like": False,
            "empty": False,
        },
        {
            "condition": "baseline",
            "label": 0,
            "refusal_phrase": False,
            "tool_call_like": False,
            "empty": False,
        },
        {
            "condition": "candidate",
            "label": 0,
            "refusal_phrase": False,
            "tool_call_like": False,
            "empty": False,
        },
    ]

    summary = summarize_smoke_rows(rows)
    candidate_harmful = next(
        row for row in summary if row["condition"] == "candidate" and row["label"] == 1
    )
    candidate_safe = next(
        row for row in summary if row["condition"] == "candidate" and row["label"] == 0
    )

    assert candidate_harmful["refusal_phrase_change"] == 1.0
    assert candidate_safe["refusal_phrase_change"] == 0.0
    assert condition_name("agent_all", 2.0) == "agent_all__plus__alpha_2"
    assert condition_name("agent_all", -0.5) == "agent_all__minus__alpha_0p5"
