from task_refusal.agentalign_data import _record_to_task, _split, matched_by_category


def _row(task_id, category, pattern=("Search.web_search",), tool="search_google"):
    return {
        "id": task_id, "category": category, "pattern": list(pattern),
        "messages": [{"role": "user", "content": task_id}],
        "tools": [{"type": "function", "function": {"name": tool,
                   "parameters": {"type": "object", "properties": {}}}}],
    }


def test_non_agent_align_benign_examples_are_excluded():
    assert _record_to_task(_row("glaive-1", "benign"), 42) is None
    assert _record_to_task(_row("benign-1", "benign"), 42) is not None


def test_pattern_split_is_stable_and_outcome_independent():
    pattern = ("Search.web_search", "Communication.send_email")
    assert _split(pattern, 42) == _split(pattern, 42)
    harmful = _record_to_task(_row("harmful-1", "privacy", pattern), 42)
    benign = _record_to_task(_row("benign-1", "benign", pattern), 42)
    assert harmful.split == benign.split


def test_matching_requires_exact_exposed_tools():
    pattern = ("Search.web_search",)
    harmful = _record_to_task(_row("harmful-1", "privacy", pattern), 42)
    benign_wrong = _record_to_task(_row("benign-1", "benign", pattern, "search_bing"), 42)
    benign_right = _record_to_task(_row("benign-2", "benign", pattern), 42)
    tasks = [harmful, benign_wrong, benign_right]
    # Other categories are intentionally missing, so the preflight fails.
    try:
        matched_by_category(tasks)
    except ValueError as exc:
        assert "lacks matched tasks" in str(exc)
    else:
        raise AssertionError("Missing categories must fail closed")
