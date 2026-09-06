import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from refusal_repro.agent_env import parse_agent_action


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
