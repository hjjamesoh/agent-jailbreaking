from pathlib import Path

from task_refusal.behavior import (
    choose_behavior_candidate,
    parse_wildguard_output,
    proxy_shortlist,
)
from task_refusal.config import load_config


def _proxy(position: int, layer: int, ablation: float, steering: float, kl: float):
    return {
        "position": position,
        "position_index": position + 5,
        "layer": layer,
        "ablation_refusal_score": ablation,
        "steering_refusal_score": steering,
        "kl_divergence": kl,
    }


def _behavior(alpha: float, *, feasible: bool, harmful: float, refusal: float, violation=0.0):
    return {
        "alpha": alpha,
        "feasible": feasible,
        "constraint_violation": violation,
        "harmful_response_harmful_rate": harmful,
        "harmful_refusal_rate": refusal,
        "safe_refusal_increase": 0.01,
        "safe_nll_increase": 0.02,
        "ablation_refusal_score": -1.0,
    }


def test_parse_wildguard_output() -> None:
    result = parse_wildguard_output(
        "Harmful request: yes\nResponse refusal: no\nHarmful response: yes"
    )
    assert result["prompt_harmful"] is True
    assert result["response_refusal"] is False
    assert result["response_harmful"] is True
    assert result["parse_error"] is False


def test_parse_wildguard_output_rejects_incomplete_result() -> None:
    result = parse_wildguard_output("Harmful request: yes")
    assert result["parse_error"] is True
    assert result["response_refusal"] is None


def test_proxy_shortlist_is_unique_and_metric_diverse() -> None:
    rows = [
        _proxy(-5, 0, -3.0, 0.0, 0.3),
        _proxy(-4, 1, -1.0, 5.0, 0.2),
        _proxy(-3, 2, -2.0, 1.0, 0.01),
        _proxy(-2, 3, 0.0, 0.5, 0.1),
    ]
    selected = proxy_shortlist(rows, 3)
    keys = {(row["position"], row["layer"]) for row in selected}
    assert len(selected) == len(keys) == 3
    assert {(-5, 0), (-4, 1), (-3, 2)} == keys


def test_behavior_choice_uses_only_feasible_candidates() -> None:
    unsafe_best = _behavior(8.0, feasible=False, harmful=0.0, refusal=1.0, violation=0.1)
    feasible = _behavior(2.0, feasible=True, harmful=0.2, refusal=0.8)
    selected, fallback = choose_behavior_candidate([unsafe_best, feasible])
    assert selected is feasible
    assert fallback is False


def test_behavior_choice_reports_constraint_fallback() -> None:
    high_violation = _behavior(
        8.0, feasible=False, harmful=0.0, refusal=1.0, violation=0.2
    )
    low_violation = _behavior(
        2.0, feasible=False, harmful=0.5, refusal=0.5, violation=0.05
    )
    selected, fallback = choose_behavior_candidate([high_violation, low_violation])
    assert selected is low_violation
    assert fallback is True


def test_main_config_has_full_position_and_alpha_search() -> None:
    config = load_config(Path("configs/main_agent_llama31.yaml"))
    assert config.method.positions == (-5, -4, -3, -2, -1)
    assert config.selection.shortlist_size == 8
    assert config.selection.alpha_grid == (0.5, 1.0, 2.0, 4.0, 8.0)
    assert config.output_dir.name.endswith("_v1")
