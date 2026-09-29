import pytest

from task_refusal.causal_selection import (
    CandidateEvidence,
    CausalThresholds,
    assess_candidate,
    paired_delta,
    select_candidate,
)


LIMITS = CausalThresholds(4, 0.25, 0.25, 0.25, 0.05, 0.05, 0.05)


def _delta(base, changed):
    ids = [str(i) for i in range(len(base))]
    return paired_delta(dict(zip(ids, base)), dict(zip(ids, changed)))


def _evidence(name="candidate", *, benign_success=None, benign_tool_use=None):
    return CandidateEvidence(
        name,
        ablation_harmful_refusal=_delta([1, 1, 1, 1], [0, 0, 1, 1]),
        addition_harmful_refusal=_delta([0, 0, 0, 0], [1, 1, 0, 0]),
        addition_harmful_unsafe_action=_delta([1, 1, 1, 1], [0, 0, 1, 1]),
        addition_benign_refusal=_delta([0, 0, 0, 0], [0, 0, 0, 0]),
        addition_benign_success=_delta([1, 1, 1, 1], benign_success or [1, 1, 1, 1]),
        addition_benign_tool_use=_delta([1, 1, 1, 1], benign_tool_use or [1, 1, 1, 1]),
        addition_benign_raw_tool_use=_delta([1, 1, 1, 1], [1, 1, 1, 1]),
    )


def test_paired_delta_requires_same_tasks_and_boolean_judgments():
    with pytest.raises(ValueError):
        paired_delta({"a": True}, {"b": False})
    with pytest.raises(TypeError):
        paired_delta({"a": 1}, {"a": False})


def test_causal_candidate_passes_only_all_required_effects():
    result = assess_candidate(_evidence(), LIMITS)
    assert result.eligible
    assert result.reasons == ()


def test_benign_success_loss_is_disqualifying():
    result = assess_candidate(_evidence(benign_success=[1, 1, 1, 0]), LIMITS)
    assert not result.eligible
    assert "benign_success_drop_exceeded" in result.reasons


def test_benign_tool_use_loss_is_disqualifying():
    result = assess_candidate(_evidence(benign_tool_use=[1, 1, 1, 0]), LIMITS)
    assert not result.eligible
    assert "benign_tool_use_drop_exceeded" in result.reasons


def test_raw_tool_use_loss_is_disqualifying_even_with_adapter_success():
    evidence = _evidence()
    evidence = CandidateEvidence(
        **{**evidence.__dict__,
           "addition_benign_raw_tool_use": _delta(
               [1, 1, 1, 1], [1, 1, 1, 0]
           )}
    )
    result = assess_candidate(evidence, LIMITS)
    assert not result.eligible
    assert "benign_raw_tool_use_drop_exceeded" in result.reasons


def test_no_fallback_when_every_candidate_fails():
    selected, decisions = select_candidate(
        [_evidence(benign_success=[1, 1, 1, 0])], LIMITS
    )
    assert selected is None
    assert len(decisions) == 1
    assert not decisions[0].eligible
