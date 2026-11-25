"""Regression: marker token IDs depend on context; a shared token can contain B too."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT.parent / "cast_scope_v1"))
from engine import A_STOP, OracleDecode, raw_decode, split_before_marker  # noqa: E402


class ChunkTokens:
    eos_token_id = 99

    def __init__(self, middle="END", suffix="_A"):
        self.chunks = {0: "prefix", 1: "Safe", 2: ".\n[[", 3: middle, 4: suffix,
                       5: "]]\n[[B]]\nWRONG B MUST BE REMOVED", 6: ".\n",
                       70: "[[END_A]]", 71: "[[B]]"}

    def encode(self, text, add_special_tokens=False):
        return {".\n": [6], "[[END_A]]": [70], "[[B]]": [71]}.get(text, [])

    def decode(self, ids, **kwargs):
        return "".join(self.chunks[i] for i in ids)


class ScriptedDecode(OracleDecode):
    def __init__(self, tok):
        import torch
        self.torch = torch
        model = SimpleNamespace(device=torch.device("cpu"), generation_config=SimpleNamespace(eos_token_id=99))
        super().__init__(model, tok, None)
        self.script = [1, 2, 3, 4, 5, 99]
        self.position = 0
        self.ids = [0]
        self.set_next()

    def set_next(self):
        self.logits = self.torch.full((100,), -100.0)
        self.logits[self.script[min(self.position, len(self.script) - 1)]] = 100.0

    def feed(self, ids, enabled=False):
        self.ids.extend(ids)
        self.position += 1
        self.set_next()

    def crop(self, length):
        self.ids = self.ids[:length]
        self.logits = None


class BoundaryTests(unittest.TestCase):
    def test_contextual_marker_not_equal_standalone_ids(self):
        tok = ChunkTokens()
        self.assertNotEqual([3, 4, 5], tok.encode("[[END_A]]"))
        engine = ScriptedDecode(tok)
        result = engine.generate(20, True, "A")
        self.assertEqual(result["stop_reason"], "section_marker")
        self.assertEqual(result["text"], "Safe.\n")
        self.assertEqual(result["token_ids"], [1, 6])
        self.assertEqual(engine.ids, [0, 1, 6])
        self.assertNotIn("[[B]]", raw_decode(tok, engine.ids))
        self.assertEqual(result["boundary_event"]["reencoded_A_or_B_tail"], ".\n")

    def test_case_variant_and_direct_B_header_both_stop_A(self):
        for middle, suffix in [("end", "_a"), ("B", "")]:
            tok = ChunkTokens(middle, suffix)
            result = ScriptedDecode(tok).generate(20, True, "A")
            self.assertEqual(result["text"], "Safe.\n")
            self.assertEqual(result["stop_reason"], "section_marker")

    def test_B_ends_before_fabricated_followup(self):
        result = ScriptedDecode(ChunkTokens("END", "_B")).generate(20, False, "B")
        self.assertEqual(result["text"], "Safe.\n")
        self.assertNotIn("WRONG", result["text"])

    def test_split_preserves_pre_marker_punctuation(self):
        tok = ChunkTokens()
        ids = [1, 2, 3, 4, 5]
        match = A_STOP.search(raw_decode(tok, ids))
        keep, tail, text = split_before_marker(tok, ids, match.start())
        self.assertEqual(raw_decode(tok, ids[:keep] + tail), "Safe.\n")
        self.assertEqual(text, ".\n")

    def test_actual_llama_cache_removes_marker_and_B_states(self):
        import numpy as np
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from core import Steering
        torch.set_num_threads(1)
        torch.manual_seed(11)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=16,
            intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
            num_key_value_heads=2, eos_token_id=63)).eval()
        tok = ChunkTokens()
        s = Steering(model, {1: np.ones(16)}, [1], 0.3)
        try:
            polluted = OracleDecode(model, tok, s)
            polluted.feed([0], False)
            polluted.feed([1, 2, 3, 4, 5], True)
            keep, tail, _ = split_before_marker(tok, [1, 2, 3, 4, 5], 6)
            polluted.crop(1 + keep)
            polluted.feed(tail, True)
            clean = OracleDecode(model, tok, s)
            clean.feed([0], False)
            clean.feed([1], True)
            clean.feed([6], True)
            self.assertEqual(polluted.ids, clean.ids)
            torch.testing.assert_close(polluted.logits, clean.logits, atol=1e-6, rtol=1e-5)
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
