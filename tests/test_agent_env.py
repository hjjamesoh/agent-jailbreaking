import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from refusal_repro.agent_env import (
    classify_agent_response,
    parse_agent_action,
    truncate_to_first_agent_action,
)


def test_parse_agent_action_tool_call():
    parsed = parse_agent_action(
        "Thought: I should inspect the request.\n"
        "Action: lookup_policy\n"
        "Action Input: summarize the safety issue"
    )
    assert parsed["type"] == "tool"
    assert parsed["tool_name"] == "lookup_policy"
    assert parsed["tool_input"] == "summarize the safety issue"


def test_parse_agent_action_final_answer():
    parsed = parse_agent_action("Final: I can help with a safe alternative.")
    assert parsed["type"] == "final"
    assert parsed["final_answer"] == "I can help with a safe alternative."


def test_parse_agent_action_force_finish_rejects_non_finish_tool():
    parsed = parse_agent_action(
        "Thought: I should inspect again.\n"
        "Action: lookup_policy\n"
        "Action Input: safety issue",
        force_finish=True,
    )
    assert parsed["type"] == "invalid_tool"
    assert parsed["tool_name"] == "lookup_policy"
    assert parsed["allowed_tools"] == ["finish"]


def test_parse_agent_action_force_finish_allows_finish():
    parsed = parse_agent_action(
        "Thought: I should answer now.\n"
        "Action: finish\n"
        "Action Input: Here is the final answer.",
        force_finish=True,
    )
    assert parsed["type"] == "final"
    assert parsed["tool_name"] == "finish"
    assert parsed["final_answer"] == "Here is the final answer."


def test_truncate_to_first_agent_action():
    text = (
        "Thought: first\n"
        "Action: lookup_policy\n"
        "Action Input: safety issue\n\n"
        "Thought: second\n"
        "Action: calculator\n"
        "Action Input: 2 + 2"
    )
    clipped = truncate_to_first_agent_action(text)
    assert "Action Input: safety issue" in clipped
    assert "calculator" not in clipped


def test_classify_agent_response_safe_alternative():
    assert classify_agent_response(
        "The request is safety-sensitive. I can provide a safe alternative."
    ) == "safe_alternative"


def test_classify_agent_response_recommendation_as_safe_alternative():
    assert classify_agent_response(
        "I recommend writing about animal welfare and ethical treatment instead."
    ) == "safe_alternative"


def test_classify_agent_response_unfinished_tool_call():
    assert classify_agent_response(
        "Thought: I should inspect the request.\n"
        "Action: lookup_policy\n"
        "Action Input: harmful request"
    ) == "unfinished_tool_call"
