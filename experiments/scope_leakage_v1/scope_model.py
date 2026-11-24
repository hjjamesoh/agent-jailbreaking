"""Cached section-wise decoding with explicit prediction-position intervention.

No agent, tools, learned gate, or automatic safety judge. Importing this module
does not require torch; the CPU data/review commands stay lightweight.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import inspect

from scope_data import SYSTEM, SELECTIVE_SYSTEM, active_for, digest


class ScopeModel:
    def __init__(self, name, revision="main", local_only=False):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("GPU required. Set CUDA_VISIBLE_DEVICES to one available GPU.")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(
            name, revision=revision, local_files_only=local_only
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            revision=revision,
            local_files_only=local_only,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        ).eval()
        if self.model.config.model_type not in {"qwen2", "llama"}:
            raise ValueError("Only Qwen2 and Llama decoder layouts are supported by this pilot")
        self.blocks = self.model.model.layers
        self.device = self.model.get_input_embeddings().weight.device
        self.eos = self.model.generation_config.eos_token_id
        if self.eos is None:
            self.eos = self.tok.eos_token_id
        self.eos = {self.eos} if isinstance(self.eos, int) else set(self.eos)
        self.unexpected_special = set(self.tok.all_special_ids) - self.eos
        self.keep_logits = "logits_to_keep" in inspect.signature(self.model.forward).parameters
        self.fingerprint = {
            "model": name,
            "requested_revision": revision,
            "model_commit": getattr(self.model.config, "_commit_hash", None),
            "model_config_sha256": digest(self.model.config.to_dict()),
            "tokenizer_sha256": hashlib.sha256(
                self.tok.backend_tokenizer.to_str().encode()
            ).hexdigest(),
            "chat_template_sha256": digest(self.tok.chat_template),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "dtype": "bfloat16",
            "attention": "sdpa",
            "hook": "resid_pre",
            "extraction_position": "last token of exact assistant generation header (-1)",
            "intervention_position": "last input position predicting next body token only",
        }

    def prompt_ids(self, prompt, selective=False):
        return self.tok.apply_chat_template(
            [
                {"role": "system", "content": SELECTIVE_SYSTEM if selective else SYSTEM},
                {"role": "user", "content": prompt},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )

    def forward(self, ids, past=None):
        torch = self.torch
        if not ids:
            raise ValueError("Cannot forward an empty input")
        values = torch.tensor([ids], device=self.device)
        past_length = past.get_seq_length() if past is not None else 0
        kwargs = {
            "input_ids": values,
            "past_key_values": past,
            "use_cache": True,
            "attention_mask": torch.ones(
                (1, past_length + len(ids)), dtype=torch.long, device=self.device
            ),
        }
        if self.keep_logits:
            kwargs["logits_to_keep"] = 1
        return self.model(**kwargs)

    def capture(self, ids, layers):
        captured = {}

        def hook_for(layer):
            def capture_hook(module, inputs, kwargs):
                h = inputs[0] if inputs else kwargs["hidden_states"]
                captured[layer] = h[0, -1].detach().float().cpu()

            return capture_hook

        handles = [
            self.blocks[layer].register_forward_pre_hook(hook_for(layer), with_kwargs=True)
            for layer in layers
        ]
        try:
            with self.torch.inference_mode():
                self.forward(ids)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    @contextmanager
    def addition(self, layer, vector, coefficient):
        if layer is None or coefficient == 0:
            yield {"calls": 0, "positions": 0}
            return
        audit = {"calls": 0, "positions": 0}

        def hook(module, inputs, kwargs):
            h = inputs[0] if inputs else kwargs["hidden_states"]
            updated = h.clone()
            # The prefix may contain many tokens (prefill/forced header). Only
            # the final prediction site is changed, never all prompt positions.
            updated[:, -1, :] += (coefficient * vector).to(h)
            audit["calls"] += 1
            audit["positions"] += h.shape[0]
            if inputs:
                return (updated, *inputs[1:]), kwargs
            return inputs, {**kwargs, "hidden_states": updated}

        handle = self.blocks[layer].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            yield audit
        finally:
            handle.remove()

    def response(self, prompt, parts, arm, layer, vector, coefficient, max_tokens):
        """Force headers, not answers. Retain one shared assistant transcript.

        At every prediction, process only tokens not yet in the cache. An EOS
        terminates the current body; it is not inserted into the shared assistant
        transcript. A token cap is a censored observation, never a clean refusal.
        """
        torch = self.torch
        ids = self.prompt_ids(prompt, selective=arm == "prompt")
        initial_ids = list(ids)
        past, processed = None, 0
        output = []
        with torch.inference_mode():
            for index, part in enumerate(parts):
                header = f"\n\n### Response {index + 1}\n"
                ids.extend(self.tok.encode(header, add_special_tokens=False))
                active = active_for(arm, part["id"])
                body = []
                stop = "token_cap"
                first_logits = None
                with self.addition(layer, vector, coefficient if active else 0) as audit:
                    for _ in range(max_tokens):
                        result = self.forward(ids[processed:], past)
                        processed = len(ids)
                        past = result.past_key_values
                        logits = result.logits[0, -1].float()
                        if first_logits is None:
                            top = torch.topk(logits, 5)
                            first_logits = {
                                "ids": top.indices.tolist(),
                                "logits": top.values.tolist(),
                            }
                        token = int(logits.argmax())
                        if token in self.eos:
                            stop = "eos"
                            break
                        if token in self.unexpected_special:
                            stop = "unexpected_special"
                            break
                        body.append(token)
                        ids.append(token)
                        text = self.tok.decode(body, skip_special_tokens=False)
                        if "<END_PART>" in text:
                            stop = "end_marker"
                            break
                        if "### Response" in text:
                            stop = "unexpected_header"
                            break
                text = self.tok.decode(body, skip_special_tokens=True).split("<END_PART>")[0]
                text = text.split("### Response")[0].strip()
                output.append(
                    {
                        **part,
                        "position": index,
                        "text": text,
                        "token_ids": body,
                        "stop_reason": stop,
                        "generated_tokens": len(body),
                        "active": active,
                        "hook_audit": dict(audit),
                        "first_prediction_top5": first_logits,
                        "truncated": stop == "token_cap",
                        "format_error": stop in {"unexpected_header", "unexpected_special"}
                        or not text,
                    }
                )
                # Replay EXACT token IDs, not retokenized text. This removes the
                # hidden cache edits but retains generated refusal wording.
                if part["id"] == "target" and arm in {"scoped_replay", "baseline_replay"}:
                    del past
                    replay = self.forward(ids)
                    past = replay.past_key_values
                    processed = len(ids)
                    output[-1]["replayed_without_intervention"] = True
        return {
            "parts": output,
            "prompt_token_ids": initial_ids,
            "transcript_token_ids": ids,
            "prompt_suffix_tokens": self.tok.convert_ids_to_tokens(initial_ids[-8:]),
        }
