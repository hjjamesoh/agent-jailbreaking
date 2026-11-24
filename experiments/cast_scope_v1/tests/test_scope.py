"""CPU mechanistic checks; no model weights or harmful samples downloaded."""
import sys
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import Decode, Steering, diagnostics, refusal_score, segment  # noqa: E402


class Tokens:
    eos_token_id = 63

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 50 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "user fixture assistant"


class ScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cls.model = LlamaForCausalLM(LlamaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=48,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=512, eos_token_id=63,
            attn_implementation="eager",
        )).eval()
        cls.tok = Tokens()
        cls.vector = {1: torch.randn(32).numpy()}

    def setUp(self):
        self.steering = Steering(self.model, self.vector, [1], 0.7)

    def tearDown(self):
        self.steering.close()

    def test_unsteered_incremental_and_replay(self):
        import torch
        engine = Decode(self.model, self.tok, self.steering)
        engine.feed([2, 4, 8])
        engine.feed([9])
        other = Decode(self.model, self.tok, self.steering)
        other.feed([2, 4, 8, 9])
        torch.testing.assert_close(engine.logits, other.logits, atol=1e-6, rtol=1e-5)
        self.assertEqual(self.steering.calls, 0)

    def test_local_cache_replay_changes_logits_without_hooks_on_B(self):
        baseline = segment(self.model, self.tok, self.steering, [2, 4, 8], 6, 5,
                           False, False)
        baseline_replay = segment(self.model, self.tok, self.steering, [2, 4, 8], 6, 5,
                                  False, False, True, baseline["a"]["token_ids"])
        self.assertEqual(baseline["b"]["token_ids"], baseline_replay["b"]["token_ids"])
        local = segment(self.model, self.tok, self.steering, [2, 4, 8], 6, 5, True, False)
        replay = segment(self.model, self.tok, self.steering, [2, 4, 8], 6, 5, True, False,
                         True, local["a"]["token_ids"])
        self.assertEqual(local["prefix_ids"], replay["prefix_ids"])
        self.assertGreater(float((local["boundary_logits"] - replay["boundary_logits"]).abs().max()), 1e-5)
        self.assertFalse(self.steering.enabled)

    def test_steering_gate_disabled_leaves_cache_and_logit_unchanged(self):
        import torch
        engine = Decode(self.model, self.tok, self.steering)
        engine.feed([2, 4, 8], False)
        self.assertEqual(self.steering.calls, 0)
        engine.feed([5], True)
        calls = self.steering.calls
        self.assertEqual(calls, 1)
        engine.feed([6, 7], False)
        self.assertEqual(self.steering.calls, calls)
        retained = engine.logits.clone()
        replay = Decode(self.model, self.tok, self.steering)
        replay.feed(list(engine.ids), False)
        self.assertGreater(float((retained - replay.logits).abs().max()), 1e-5)
        self.assertTrue(torch.isfinite(retained).all())

    def test_crop_removes_marker_state(self):
        import torch
        a = Decode(self.model, self.tok, self.steering)
        a.feed([2, 4, 8, 9])
        a.crop(3)
        a.feed([10])
        b = Decode(self.model, self.tok, self.steering)
        b.feed([2, 4, 8, 10])
        self.assertEqual(a.ids, b.ids)
        torch.testing.assert_close(a.logits, b.logits, atol=1e-6, rtol=1e-5)

    def test_primary_refusal_and_quality_are_independent(self):
        self.assertEqual(refusal_score("I can't help. " + "sorry " * 40), 1)
        self.assertGreater(diagnostics("I can't help. " + "sorry " * 40)["max_identical_word_run"], 30)
        self.assertEqual(refusal_score("Here is the answer."), 0)

    def test_pilot_checkpoint_report_and_resume(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("scope_runner", ROOT / "run.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "records").mkdir()
            (run / "settings.json").write_text(json.dumps({"stage": "pilot"}))
            args = SimpleNamespace(alpha=0.2, a_max_tokens=5, b_max_tokens=6)
            pairs = [({"id": "h-fixture", "original_prompt": "A fixture"},
                      {"id": "b-fixture", "original_prompt": "Compute 1+1", "expected": "2"})]
            with patch.object(runner.shared, "LAYERS", [1]), redirect_stdout(io.StringIO()):
                runner.pilot(args, self.model, self.tok, self.vector, run, pairs)
                runner.report(run)
                summary = json.loads((run / "summary.json").read_text())
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["baseline_replay_token_mismatches"], 0)
                self.assertEqual(len(summary["groups"]), 5)
                self.assertEqual(len((run / "responses.jsonl").read_text().splitlines()), 5)
                before = (run / "records" / "pair_000_local_A.json").read_bytes()
                # A completed pair must resume without any model inference.
                with patch.object(self.model, "forward", side_effect=AssertionError("not resumed")):
                    runner.pilot(args, self.model, self.tok, self.vector, run, pairs)
                self.assertEqual(before, (run / "records" / "pair_000_local_A.json").read_bytes())
                # An interruption after records but before parity must be repaired.
                (run / "parity" / "pair_000.json").unlink()
                runner.pilot(args, self.model, self.tok, self.vector, run, pairs)
                self.assertTrue((run / "parity" / "pair_000.json").is_file())


if __name__ == "__main__":
    unittest.main()
