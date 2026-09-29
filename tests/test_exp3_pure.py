import json
import tempfile
import unittest
from pathlib import Path

from task_refusal.agentalign_data import _record_to_task, balanced_tool_pairs
from task_refusal.causal_selection import (
    CandidateEvidence, CausalThresholds, paired_delta, select_candidate,
)
from task_refusal.exp3_review import render_review_file
from task_refusal.exp3_stats import holm_adjust, paired_effect, paired_sign_pvalue
from task_refusal.agent_protocol import normalize_unicode, parse_tool_call
from task_refusal.agent_simulator import _validate_arguments
from task_refusal.llama31_prompt import render_llama31_tool_chat


def row(task_id, category, tool):
    return _record_to_task({
        "id": task_id, "category": category, "pattern": [tool],
        "messages": [{"role": "user", "content": task_id}],
        "tools": [{"type": "function", "function": {"name": tool,
                    "parameters": {"type": "object", "properties": {}}}}],
    }, 42)


class Exp3PureTests(unittest.TestCase):
    def test_surrogate_unicode_is_repaired_before_persistence(self):
        pair = chr(0xD83D) + chr(0xDE80)
        self.assertEqual(normalize_unicode(pair), "🚀")
        self.assertEqual(normalize_unicode(chr(0xD83D)), "�")
        call = parse_tool_call(
            '{"name":"post_instagram","arguments":{"caption":"\\ud83d"}}'
        )
        self.assertEqual(call[1]["caption"], "�")

    @staticmethod
    def _schema(property_schema):
        return {"function": {"parameters": {
            "type": "object",
            "properties": {"value": property_schema},
            "required": ["value"],
        }}}

    def test_tool_argument_union_types(self):
        schema = self._schema({"type": ["string", "integer"]})
        _validate_arguments(schema, {"value": "123"})
        _validate_arguments(schema, {"value": 123})
        with self.assertRaisesRegex(ValueError, "wrong type"):
            _validate_arguments(schema, {"value": []})

        nullable = self._schema({"type": ["string", "null"]})
        _validate_arguments(nullable, {"value": None})

    def test_json_number_does_not_accept_boolean(self):
        schema = self._schema({"type": ["integer", "null"]})
        with self.assertRaisesRegex(ValueError, "must not be boolean"):
            _validate_arguments(schema, {"value": True})

    def test_llama31_renderer_handles_emoji_and_tool_history(self):
        messages = [
            {"role": "system", "content": "safe system"},
            {"role": "user", "content": "Post this 🚀💰"},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {
                "name": "post_instagram",
                "arguments": {"media_urls": ["https://example.com/a.mp4"],
                              "is_sensitive": True},
            }}]},
            {"role": "tool", "name": "post_instagram",
             "content": '{"status": "posted"}'},
        ]
        tools = [{"name": "post_instagram", "parameters": {
            "type": "object", "properties": {
                "media_urls": {"type": "array"},
                "is_sensitive": {"type": "boolean", "default": False},
            }}}]
        rendered = render_llama31_tool_chat(
            messages, tools, bos_token="<|begin_of_text|>",
        )
        self.assertIsInstance(rendered, str)
        self.assertIn("Post this 🚀💰", rendered)
        self.assertIn('"name": "post_instagram"', rendered)
        self.assertIn("<|start_header_id|>ipython<|end_header_id|>", rendered)
        self.assertTrue(rendered.endswith(
            "<|start_header_id|>assistant<|end_header_id|>\n\n"))

    def test_strict_tool_parser(self):
        self.assertEqual(parse_tool_call('{"name":"search_google","arguments":{"query":"x"}}'),
                         ("search_google", {"query": "x"}))
        self.assertEqual(parse_tool_call('to=search_google {"query":"x"}'),
                         ("search_google", {"query": "x"}))
        self.assertIsNone(parse_tool_call('please run search_google for x'))
        self.assertIsNone(parse_tool_call('{"name":"__import__","arguments":{}}'))

    def test_parser_executes_only_first_complete_speculative_call(self):
        response = (
            '{"name":"search_google","parameters":{"query":"x"}}; '
            '{"name":"send_email","parameters":{"to":"later"}}'
        )
        self.assertEqual(parse_tool_call(response),
                         ("search_google", {"query": "x"}))
        # A truncated future call must not invalidate the completed first call.
        self.assertEqual(parse_tool_call(response[:-8]),
                         ("search_google", {"query": "x"}))
        # JSON embedded after natural-language text is not a tool action.
        self.assertIsNone(parse_tool_call(
            'I suggest {"name":"search_google","parameters":{"query":"x"}}'))

    def test_exact_tool_balancing(self):
        harmful = [row("harmful-1", "privacy", "search_google"),
                   row("harmful-2", "privacy", "search_google"),
                   row("harmful-3", "privacy", "search_bing")]
        benign = [row("benign-1", "benign", "search_google"),
                  row("benign-2", "benign", "search_bing")]
        h, b = balanced_tool_pairs(harmful, benign, max_per_side=10, seed=42)
        self.assertEqual(len(h), 2)
        self.assertEqual(sorted(x.tool_names for x in h),
                         sorted(x.tool_names for x in b))

    def test_no_invalid_direction_fallback(self):
        ids = [str(i) for i in range(4)]
        delta = lambda a, b: paired_delta(dict(zip(ids, a)), dict(zip(ids, b)))
        evidence = CandidateEvidence(
            "x", delta([True]*4, [True]*4),
            delta([False]*4, [True, False, False, False]),
            delta([True]*4, [False, True, True, True]),
            delta([False]*4, [False]*4),
            delta([True]*4, [True]*4),
            delta([True]*4, [True]*4),
        )
        selected, decisions = select_candidate([evidence], CausalThresholds(
            4, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05))
        self.assertIsNone(selected)
        self.assertIn("ablation_did_not_reduce_harmful_refusal", decisions[0].reasons)

    def test_tool_use_loss_blocks_otherwise_valid_direction(self):
        ids = [str(i) for i in range(4)]
        delta = lambda a, b: paired_delta(dict(zip(ids, a)), dict(zip(ids, b)))
        evidence = CandidateEvidence(
            "x", delta([True]*4, [False, True, True, True]),
            delta([False]*4, [True, False, False, False]),
            delta([True]*4, [False, True, True, True]),
            delta([False]*4, [False]*4),
            delta([True]*4, [True]*4),
            delta([True]*4, [False, True, True, True]),
        )
        selected, decisions = select_candidate([evidence], CausalThresholds(
            4, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05))
        self.assertIsNone(selected)
        self.assertIn("benign_tool_use_drop_exceeded", decisions[0].reasons)

    def test_paired_statistics_and_holm(self):
        a = {str(i): False for i in range(20)}
        b = {str(i): i < 5 for i in range(20)}
        effect = paired_effect(a, b, resamples=500)
        self.assertEqual(effect["n"], 20)
        self.assertEqual(effect["delta"], 0.25)
        self.assertLessEqual(effect["ci95_low"], effect["ci95_high"])
        self.assertEqual(paired_sign_pvalue(a, a), 1)
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04})
        self.assertEqual(adjusted["a"], 0.03)
        self.assertEqual(adjusted["b"], 0.06)
        self.assertEqual(adjusted["c"], 0.06)

    def test_review_escapes_model_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "review.jsonl"
            source.write_text(json.dumps({
                "task_id": "x", "prompt": "<script>alert(1)</script>",
                "comparisons": {"baseline": {"rollout": {"steps": [{"assistant": "hi"}]},
                                              "judgment": None}},
            }) + "\n", encoding="utf-8")
            output = render_review_file(source)
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("&lt;script&gt;", rendered)
            self.assertNotIn("<script>", rendered)


if __name__ == "__main__":
    unittest.main()
