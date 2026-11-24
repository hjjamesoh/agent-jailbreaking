"""Offline real-transformer tests; skipped if GPU runtime dependencies are absent."""

import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scope_model import ScopeModel  # noqa: E402

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers"))


class ByteTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode("ascii"))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(v)) for v in ids)

    def convert_ids_to_tokens(self, ids):
        return [str(v) for v in ids]

    def apply_chat_template(self, messages, **kwargs):
        return self.encode("USER:" + messages[-1]["content"] + "\nASSISTANT:\n")


@unittest.skipUnless(AVAILABLE, "torch/transformers not installed; run again in server environment")
class CacheTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(42)
        config = Qwen2Config(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=2048,
            attention_dropout=0.0,
        )
        self.h = ScopeModel.__new__(ScopeModel)
        self.h.torch = torch
        self.h.model = Qwen2ForCausalLM(config).eval()
        self.h.tok = ByteTokenizer()
        self.h.blocks = self.h.model.model.layers
        self.h.device = torch.device("cpu")
        self.h.eos = set()  # fixed body length for cache boundary assertions
        self.h.unexpected_special = set()
        self.h.keep_logits = False
        self.vector = torch.randn(32)
        self.vector /= self.vector.norm()
        self.parts = [
            {"id": "safe0", "prompt": "One", "expected": "1"},
            {"id": "target", "prompt": "Two", "expected": None},
            {"id": "safe1", "prompt": "Three", "expected": "3"},
        ]

    def test_cached_forward_matches_full_forward(self):
        import torch

        with torch.inference_mode():
            cached = self.h.forward([10, 20, 30])
            incremental = self.h.forward([40], cached.past_key_values).logits[0, -1]
            full = self.h.forward([10, 20, 30, 40]).logits[0, -1]
        torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-4)

    def test_only_prediction_site_is_modified(self):
        import torch

        plain = self.h.capture([10, 20, 30], [1])[1]
        with self.h.addition(1, self.vector, 2.0) as audit:
            edited = self.h.capture([10, 20, 30], [1])[1]
        torch.testing.assert_close(edited, plain + 2 * self.vector)
        self.assertEqual(audit, {"calls": 1, "positions": 1})
        torch.testing.assert_close(self.h.capture([10, 20, 30], [1])[1], plain)

    def test_gate_and_hooks_are_removed_at_section_boundaries(self):
        output = self.h.response("Test", self.parts, "scoped", 1, self.vector, 1.0, 3)
        self.assertEqual([p["active"] for p in output["parts"]], [False, True, False])
        self.assertEqual([p["hook_audit"]["calls"] for p in output["parts"]], [0, 3, 0])
        self.assertEqual(len(self.h.blocks[1]._forward_pre_hooks), 0)
        self.assertTrue(all(p["truncated"] for p in output["parts"]))

    def test_zero_addition_and_baseline_replay_preserve_transcript(self):
        base = self.h.response("Test", self.parts, "baseline", None, None, 0, 3)
        zero = self.h.response("Test", self.parts, "scoped", 1, self.vector, 0, 3)
        replay = self.h.response("Test", self.parts, "baseline_replay", None, None, 0, 3)
        self.assertEqual(base["transcript_token_ids"], zero["transcript_token_ids"])
        self.assertEqual(base["transcript_token_ids"], replay["transcript_token_ids"])

    def test_scoped_replay_keeps_exact_prefix_through_target(self):
        original = self.h.response("Test", self.parts, "scoped", 1, self.vector, 1.0, 3)
        replay = self.h.response("Test", self.parts, "scoped_replay", 1, self.vector, 1.0, 3)
        for i in (0, 1):
            self.assertEqual(original["parts"][i]["token_ids"], replay["parts"][i]["token_ids"])
        self.assertTrue(replay["parts"][1]["replayed_without_intervention"])


if __name__ == "__main__":
    unittest.main()
