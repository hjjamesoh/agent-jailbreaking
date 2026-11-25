"""Tiny random Llama verifies scope switching and checkpoint behavior; no downloads."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT.parent / "cast_scope_v1"))
from core import Steering  # noqa: E402
from engine import run_sections  # noqa: E402

spec = importlib.util.spec_from_file_location("oracle_runner", ROOT / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Tokens:
    eos_token_id = 63

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 50 for c in text]

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        return " ".join(map(str, ids))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "user fixture assistant"


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cls.model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32,
            intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, eos_token_id=63, max_position_embeddings=512)).eval()
        cls.tok = Tokens()
        cls.directions = {1: np.random.default_rng(17).normal(size=32)}

    def test_same_A_and_zero_B_hooks_in_oracle(self):
        s = Steering(self.model, self.directions, [1], 0.7)
        try:
            global_row = run_sections(self.model, self.tok, s, [2, 4, 8], 8, 6, True, True)
            local = run_sections(self.model, self.tok, s, [2, 4, 8], 8, 6, True, False,
                                 reference_a=global_row["a"]["token_ids"])
            replay = run_sections(self.model, self.tok, s, [2, 4, 8], 8, 6, True, False,
                                  True, local["a"]["token_ids"])
            self.assertGreater(local["A_intervention_layer_calls"], 0)
            self.assertGreater(global_row["B_intervention_layer_calls"], 0)
            self.assertEqual(local["B_intervention_layer_calls"], 0)
            self.assertEqual(replay["B_intervention_layer_calls"], 0)
            self.assertEqual(local["prefix_token_ids"], replay["prefix_token_ids"])
            self.assertGreater(replay["replay_boundary_logit_max_abs_difference"], 1e-5)
        finally:
            s.close()

    def test_baseline_replay_control_and_eos_not_committed(self):
        s = Steering(self.model, self.directions, [1], 0)
        try:
            a = run_sections(self.model, self.tok, s, [2, 4, 8], 8, 6, False, False)
            b = run_sections(self.model, self.tok, s, [2, 4, 8], 8, 6, False, False,
                             True, a["a"]["token_ids"])
            self.assertEqual(a["b"]["token_ids"], b["b"]["token_ids"])
            self.assertNotIn(63, a["prefix_token_ids"])
            self.assertEqual(s.calls, 0)
        finally:
            s.close()

    def test_fixed_published_extraction_on_known_B_body(self):
        a = runner.annotate({"text": "#### John paid $297 for everything.",
                             "token_ids": [], "stop_reason": "eos", "truncated": False}, "297")
        self.assertTrue(a["flexible_correct"])
        self.assertFalse(a["strict_correct"])

    def test_generation_report_and_resume_no_forward(self):
        import pipeline
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            stage1 = base / "stage1"
            stage1.mkdir()
            (stage1 / "refusal.svec").write_text(json.dumps({"directions": {"1": self.directions[1].tolist()}}))
            out = base / "oracle"
            out.mkdir()
            common = {"pair": 0, "harmful_id": "A-fixture", "benign_id": "B-fixture",
                      "harmful_prompt": "A fixture", "benign_prompt": "Compute 1+1", "expected": "2"}
            plan = [{**common, "key": arm, "arm": arm, "alpha": alpha} for arm, alpha in [
                ("B_only", 0), ("baseline", 0), ("baseline_replay", 0),
                ("global", 0.2), ("oracle_A", 0.2), ("oracle_A_replay", 0.2)]]
            runner.shared.write_json(out / "plan.json", plan)
            args = SimpleNamespace(stage1=stage1, output=out, local_only=True,
                                   a_max_tokens=5, b_max_tokens=6)
            with patch.object(pipeline, "load_model", return_value=(self.model, self.tok, {"toy": True})), \
                 patch.object(runner.shared, "LAYERS", [1]), \
                 patch.object(runner, "tokenizer_preflight", return_value=16), redirect_stdout(io.StringIO()):
                runner.generate(args, plan, {"fingerprint": {"toy": True}})
                runner.report(args)
                results = runner.load_rows(args)
                self.assertEqual(len(results), 6)
                for row in results:
                    if row["arm"] != "global":
                        self.assertEqual(row["B_intervention_layer_calls"], 0)
                path = out / "records/oracle_A.json"
                before = path.read_bytes()
                with patch.object(self.model, "forward", side_effect=AssertionError("resume ran model")):
                    runner.generate(args, plan, {"fingerprint": {"toy": True}})
                self.assertEqual(path.read_bytes(), before)
                summary = runner.read(out / "summary.json")
                self.assertEqual(summary["n"], 6)
                self.assertTrue(all(r["A_matches_global"] for r in summary["paired"]))


if __name__ == "__main__":
    unittest.main()
