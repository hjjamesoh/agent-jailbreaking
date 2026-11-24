"""Opt-in integration tests with real, byte-verified CAST and tiny random Llama.

No model download and no GPU required. Set CAST_STAGE1_UPSTREAM to prepared vendor.
"""

import importlib.util
import os
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VENDOR = os.environ.get("CAST_STAGE1_UPSTREAM")
AVAILABLE = VENDOR and all(
    importlib.util.find_spec(m) for m in ("torch", "transformers", "sklearn", "rich", "matplotlib")
)


@unittest.skipUnless(
    AVAILABLE, "Optional real-CAST integration: set CAST_STAGE1_UPSTREAM after prepare"
)
class UpstreamTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        sys.path.insert(0, str(Path(VENDOR).resolve()))
        from activation_steering import MalleableModel, SteeringDataset, SteeringVector
        from activation_steering.config import GlobalConfig

        GlobalConfig.set_verbose(False)
        torch.set_num_threads(1)
        torch.manual_seed(42)
        np.random.seed(42)
        vocab = {
            word: i
            for i, word in enumerate(
                [
                    "[PAD]",
                    "[UNK]",
                    "[BOS]",
                    "[EOS]",
                    "user",
                    "assistant",
                    "hello",
                    "why",
                    "sorry",
                    "no",
                    "sure",
                    "yes",
                    "please",
                    "test",
                ]
            )
        }
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        self.tok = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
            unk_token="[UNK]",
            model_input_names=["input_ids", "attention_mask"],
        )
        self.tok.padding_side = "left"
        self.tok.chat_template = (
            "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}"
            "{% if add_generation_prompt %}assistant {% endif %}"
        )
        self.model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=len(vocab),
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                eos_token_id=3,
                pad_token_id=0,
                bos_token_id=2,
            )
        ).eval()
        self.up = (MalleableModel, SteeringDataset, SteeringVector)

    def train_vector(self):
        dataset = self.up[1](
            self.tok,
            [("hello", "hello"), ("why please", "why please")],
            suffixes=[("sorry no", "sure yes"), ("no", "yes")],
        )
        self.assertEqual(len(dataset.formatted_dataset), 4)
        return self.up[2].train(
            self.model,
            self.tok,
            dataset,
            hidden_layer_ids=[0, 1],
            batch_size=2,
            method="pca_pairwise",
            accumulate_last_x_tokens="suffix-only",
        )

    def test_official_extraction_pca_and_json_roundtrip(self):
        import numpy as np
        import pipeline
        import tempfile

        vector = self.train_vector()
        for direction in vector.directions.values():
            self.assertTrue(np.isfinite(direction).all())
            self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=5)
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "vector.svec")
            pipeline.save_direction(vector, path)
            restored = self.up[2].load(path)
            np.testing.assert_allclose(restored.directions[0], vector.directions[0])

    def test_actual_steering_generation_and_reset(self):
        import pipeline
        from activation_steering.leash_layer import LeashLayer

        vector = self.train_vector()
        wrapper = self.up[0](self.model, self.tok)
        with patch.object(pipeline, "LAYERS", [0, 1]):
            baseline = pipeline.generate_one(wrapper, self.tok, "hello", None, 0.0, 4)
            edited = pipeline.generate_one(wrapper, self.tok, "hello", vector, 1.7, 4)
            baseline_after = pipeline.generate_one(wrapper, self.tok, "hello", None, 0.0, 4)
        self.assertEqual(baseline["token_ids"], baseline_after["token_ids"])
        self.assertTrue(all(v > 0 for v in edited["layer_forward_calls"].values()))
        self.assertFalse(LeashLayer.forward_calls)

    def test_pipeline_resumes_official_activation_batches_without_model_calls(self):
        import json
        import tempfile
        import activation_steering
        import pipeline

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            demo = root / "vendor/docs/demo-data"
            demo.mkdir(parents=True)
            (demo / "alpaca.json").write_text(
                json.dumps({"train": [{"question": "hello"}] * 100}), encoding="utf-8"
            )
            (demo / "behavior_refusal.json").write_text(
                json.dumps(
                    {
                        "non_compliant_responses": ["sorry no"] * 100,
                        "compliant_responses": ["sure yes"] * 100,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(pipeline, "official", return_value=activation_steering),
                patch.object(
                    pipeline, "load_model", return_value=(self.model, self.tok, {"toy": True})
                ),
                patch.object(pipeline, "LAYERS", [0, 1]),
                redirect_stdout(io.StringIO()),
            ):
                pipeline.extract(root, batch_size=4096)
                first = self.up[2].load(str(root / "refusal.svec"))
                self.assertEqual(len(list((root / "extraction").glob("*.npz"))), 5)
                (root / "refusal.svec").unlink()
                with patch.object(
                    self.model, "forward", side_effect=AssertionError("cache not used")
                ):
                    pipeline.extract(root, batch_size=4096)
                second = self.up[2].load(str(root / "refusal.svec"))
            import numpy as np

            np.testing.assert_allclose(first.directions[0], second.directions[0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
