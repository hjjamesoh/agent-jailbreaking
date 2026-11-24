"""Protocol tests and opt-in real official-CAST tiny-Llama integration."""
from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocol import (generation_metrics, length_batched_hiddens, official_harm_template,  # noqa: E402
                      sections, upstream_comparator)

spec = importlib.util.spec_from_file_location("mixed_runner", ROOT / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
VENDOR = os.environ.get("CAST_STAGE1_UPSTREAM")


class ProtocolTests(unittest.TestCase):
    def test_sections_both_orders_and_duplicate_unknown(self):
        part, _ = sections("[[B]]\nWork\n#### 2\n[[A]]\nI cannot help.")
        self.assertEqual(part["B"], "Work\n#### 2")
        self.assertEqual(part["A"], "I cannot help.")
        part, counts = sections("[[A]]\nOne\n[[A]]\nTwo\n[[B]]\n#### 2")
        self.assertIsNone(part["A"])
        self.assertEqual(counts["A"], 2)

    def test_normal_task_score_does_not_read_A_numbers(self):
        parse = runner.shared.final_number
        r = generation_metrics("[[A]]\n#### 2\n[[B]]\nI cannot help.", "mixed_AB", parse, "2")
        self.assertFalse(r["B_numeric_correct"])
        r = generation_metrics("[[A]]\nNo\n[[B]]\n#### 2", "mixed_AB", parse, "2")
        self.assertTrue(r["B_numeric_correct"])
        r = generation_metrics("I cannot help with either request.", "mixed_AB", parse, "2")
        self.assertIsNone(r["sections"]["A"])
        self.assertEqual(r["B_failure_kind"], "missing_or_duplicate_section")

    def test_harmbench_prompt_constant_is_read_not_executed(self):
        text = "raise RuntimeError('do not execute')\nLLAMA2_CLS_PROMPT = {'prompt': 'fixed {behavior} {generation}'}"
        self.assertEqual(official_harm_template(text), "fixed {behavior} {generation}")

    def test_missing_refusal_label_reported_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            row = {"key": "example", "pair": 0, "arm": "baseline", "layout": "mixed_AB",
                   "text": "No sections", "sections": {"A": None, "B": None},
                   "gate_triggered": None, "B_numeric_correct": False, "section_format_valid": False,
                   "truncated": False, "max_identical_word_run": 1}
            runner.shared.write_json(out / "records/example.json", row)
            runner.shared.write_json(out / "data.json", {"plan": [row]})
            with redirect_stdout(io.StringIO()):
                runner.report(SimpleNamespace(output=out))
            result = runner.read(out / "summary.json")["summary"][0]
            self.assertEqual(result["A_cast_refusal_labeled"], 0)
            self.assertEqual(result["harmbench_labeled"], 0)
            self.assertEqual(result["no_unsafe_and_B_correct_evaluated"], 0)


@unittest.skipUnless(VENDOR, "Set CAST_STAGE1_UPSTREAM to existing pinned vendor folder")
class OfficialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        sys.path.insert(0, str(Path(VENDOR).resolve()))
        import activation_steering
        from activation_steering.config import GlobalConfig
        GlobalConfig.set_verbose(False)
        cls.upstream = activation_steering
        torch.set_num_threads(1)
        torch.manual_seed(42)
        backend = Tokenizer(WordLevel({w: i for i, w in enumerate([
            "[PAD]", "[UNK]", "[BOS]", "[EOS]", "user", "assistant", "hello", "test", "long", "a"
        ])}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        cls.tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                         unk_token="[UNK]", eos_token="[EOS]", bos_token="[BOS]",
                                         model_input_names=["input_ids", "attention_mask"])
        cls.tok.chat_template = "{% for m in messages %}user {{m['content']}} {% endfor %}assistant"
        cls.tok.padding_side = "left"
        cls.model = LlamaForCausalLM(LlamaConfig(
            vocab_size=10, hidden_size=32, intermediate_size=48, num_hidden_layers=9,
            num_attention_heads=4, num_key_value_heads=2, eos_token_id=3, pad_token_id=0,
        )).eval()

    def test_equal_length_pooling_matches_original_batch_one_and_resumes(self):
        import numpy as np
        from activation_steering.steering_vector import batched_get_hiddens
        values = ["hello", "hello test long", "test", "hello test"]
        original = batched_get_hiddens(self.model, self.tok, values, [7], 1, "all", None)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            result = length_batched_hiddens(batched_get_hiddens, self.model, self.tok, values,
                                            [7], 4, Path(temp), runner.shared.digest)
            np.testing.assert_allclose(original[7], result[7], atol=1e-6, rtol=1e-5)
            with patch.object(self.model, "forward", side_effect=AssertionError("cache unused")):
                cached = length_batched_hiddens(batched_get_hiddens, self.model, self.tok, values,
                                                [7], 4, Path(temp), runner.shared.digest)
            np.testing.assert_array_equal(result[7], cached[7])

    def test_real_official_gate_comparison_and_reset(self):
        import numpy as np
        from activation_steering.leash_layer import LeashLayer
        refusal = self.upstream.SteeringVector("llama", {8: np.ones(32)}, {8: 1.0})
        condition = self.upstream.SteeringVector("llama", {7: np.ones(32)}, {7: 1.0})
        wrapper = self.upstream.MalleableModel(self.model, self.tok)
        row = {"arm": "cast_calibrated", "prompt": "hello test", "layout": "mixed_AB", "expected": "2"}
        with patch.object(runner.shared, "LAYERS", [8]):
            on = runner.generate(wrapper, self.tok, row, refusal, condition,
                                 {"threshold": 2, "library_comparator": upstream_comparator("<")}, 0.2, 4)
            off = runner.generate(wrapper, self.tok, row, refusal, condition,
                                  {"threshold": -2, "library_comparator": upstream_comparator("<")}, 0.2, 4)
            base = runner.generate(wrapper, self.tok, {**row, "arm": "baseline"}, refusal, condition,
                                   {}, 0.2, 4)
        self.assertTrue(on["gate_triggered"])
        self.assertFalse(off["gate_triggered"])
        self.assertEqual(off["token_ids"], base["token_ids"])
        self.assertFalse(LeashLayer.forward_calls)
        self.assertFalse(LeashLayer.condition_met)
        # Remove wrappers for other tests sharing the same tiny model.
        self.model.model.layers = __import__("torch").nn.ModuleList([layer.layer for layer in self.model.model.layers])


if __name__ == "__main__":
    unittest.main()
