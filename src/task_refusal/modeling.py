from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from task_refusal.config import ModelConfig
from task_refusal.data import Context
from task_refusal.hooks import temporary_hooks
from task_refusal.progress import ProgressTracker


class LlamaHarness:
    def __init__(self, config: ModelConfig):
        self.config = config
        dtype = getattr(torch, config.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.name,
            revision=config.revision,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            config.name,
            revision=config.revision,
            torch_dtype=dtype,
            device_map=config.device_map,
            attn_implementation=config.attn_implementation,
        )
        self.model.eval()
        self.blocks = list(self.model.model.layers)
        self.attention_modules = [block.self_attn for block in self.blocks]
        self.mlp_modules = [block.mlp for block in self.blocks]
        self.n_layers = len(self.blocks)
        self.hidden_size = int(self.model.config.hidden_size)
        self.input_device = self.model.get_input_embeddings().weight.device

    def _render(self, context: Context) -> str:
        return self.tokenizer.apply_chat_template(
            list(context),
            tokenize=False,
            add_generation_prompt=True,
        )

    def tokenize(self, contexts: Sequence[Context]):
        if not contexts:
            raise ValueError("Cannot tokenize an empty context list.")
        rendered = [self._render(context) for context in contexts]
        batch = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            add_special_tokens=False,
        )
        return batch.to(self.input_device)

    def refusal_token_ids(self, token_strings: Sequence[str]) -> Tensor:
        ids: list[int] = []
        for text in token_strings:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(
                    f"Refusal token string {text!r} maps to {len(encoded)} tokens: {encoded}"
                )
            ids.append(int(encoded[0]))
        return torch.tensor(ids, dtype=torch.long)

    def suffix_token_labels(self, context: Context, positions: Sequence[int]) -> list[str]:
        batch = self.tokenize([context])
        ids = batch.input_ids[0]
        if min(positions) < -ids.shape[0]:
            raise ValueError("A configured activation position precedes the context.")
        return [self.tokenizer.decode([int(ids[position])]) for position in positions]

    @torch.inference_mode()
    def last_logits(
        self,
        contexts: Sequence[Context],
        *,
        pre_hooks=(),
        forward_hooks=(),
        batch_size: int | None = None,
        progress_label: str | None = None,
    ) -> Tensor:
        effective_batch_size = batch_size or self.config.batch_size
        chunks: list[Tensor] = []
        progress = (
            ProgressTracker(
                "last_token_logits",
                (len(contexts) + effective_batch_size - 1) // effective_batch_size,
                {"condition": progress_label},
            )
            if progress_label is not None
            else None
        )
        for start in range(0, len(contexts), effective_batch_size):
            batch = self.tokenize(contexts[start : start + effective_batch_size])
            with temporary_hooks(pre_hooks, forward_hooks):
                logits = self.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    use_cache=False,
                ).logits[:, -1, :]
            chunks.append(logits.detach().to(device="cpu", dtype=torch.float32))
            if progress is not None:
                progress.advance(
                    item=(f"states={start}:{min(start + effective_batch_size, len(contexts))}")
                )
        if not chunks:
            raise ValueError("Cannot score an empty context list.")
        return torch.cat(chunks, dim=0)

    def refusal_scores(
        self,
        contexts: Sequence[Context],
        refusal_token_ids: Tensor,
        *,
        pre_hooks=(),
        forward_hooks=(),
        batch_size: int | None = None,
        progress_label: str | None = None,
    ) -> Tensor:
        logits = self.last_logits(
            contexts,
            pre_hooks=pre_hooks,
            forward_hooks=forward_hooks,
            batch_size=batch_size,
            progress_label=progress_label,
        )
        return refusal_log_odds(logits, refusal_token_ids)

    @torch.inference_mode()
    def generate(
        self,
        contexts: Sequence[Context],
        *,
        pre_hooks=(),
        forward_hooks=(),
        batch_size: int | None = None,
        max_new_tokens: int = 128,
        progress_label: str | None = None,
    ) -> list[str]:
        """Greedily generate complete next-agent actions with hooks active."""
        effective_batch_size = batch_size or self.config.batch_size
        outputs: list[str] = []
        progress = ProgressTracker(
            "agent_action_generation",
            (len(contexts) + effective_batch_size - 1) // effective_batch_size,
            {"condition": progress_label or "unspecified"},
        )
        for start in range(0, len(contexts), effective_batch_size):
            batch = self.tokenize(contexts[start : start + effective_batch_size])
            prompt_width = int(batch.input_ids.shape[1])
            with temporary_hooks(pre_hooks, forward_hooks):
                sequences = self.model.generate(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.model.generation_config.eos_token_id,
                    use_cache=True,
                )
            for sequence in sequences:
                generated = sequence[prompt_width:]
                outputs.append(self.tokenizer.decode(generated, skip_special_tokens=True).strip())
            progress.advance(
                item=f"states={start}:{min(start + effective_batch_size, len(contexts))}"
            )
        if len(outputs) != len(contexts):
            raise RuntimeError("Generation count does not match the number of contexts.")
        return outputs

    @torch.inference_mode()
    def reference_nll(
        self,
        contexts: Sequence[Context],
        outputs: Sequence[str],
        *,
        pre_hooks=(),
        forward_hooks=(),
        batch_size: int | None = None,
        progress_label: str | None = None,
    ) -> float:
        """Mean teacher-forced token NLL over recorded benign next actions."""
        if len(contexts) != len(outputs):
            raise ValueError("contexts and outputs must have identical lengths.")
        examples: list[dict[str, list[int]]] = []
        output_lengths: list[int] = []
        for context, output in zip(contexts, outputs):
            output_ids = self.tokenizer.encode(str(output), add_special_tokens=False)
            if not output_ids:
                continue
            prompt_ids = self.tokenizer.encode(self._render(context), add_special_tokens=False)
            combined = (prompt_ids + output_ids)[-self.config.max_length :]
            kept_output = min(len(output_ids), len(combined))
            examples.append({"input_ids": combined, "attention_mask": [1] * len(combined)})
            output_lengths.append(kept_output)
        if not examples:
            raise ValueError("No non-empty reference outputs are available for NLL evaluation.")

        effective_batch_size = batch_size or self.config.batch_size
        total_nll = 0.0
        total_tokens = 0
        progress = ProgressTracker(
            "reference_nll",
            (len(examples) + effective_batch_size - 1) // effective_batch_size,
            {"condition": progress_label or "unspecified"},
        )
        for start in range(0, len(examples), effective_batch_size):
            chunk = examples[start : start + effective_batch_size]
            lengths = output_lengths[start : start + effective_batch_size]
            batch = self.tokenizer.pad(chunk, padding=True, return_tensors="pt").to(
                self.input_device
            )
            labels = torch.full_like(batch.input_ids, -100)
            for row, output_length in enumerate(lengths):
                labels[row, -output_length:] = batch.input_ids[row, -output_length:]
            with temporary_hooks(pre_hooks, forward_hooks):
                logits = self.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    use_cache=False,
                ).logits
            target = labels[:, 1:]
            mask = target.ne(-100)
            safe_target = target.masked_fill(~mask, 0)
            token_log_probs = (
                logits[:, :-1, :]
                .float()
                .log_softmax(dim=-1)
                .gather(-1, safe_target.unsqueeze(-1))
                .squeeze(-1)
            )
            total_nll += float((-token_log_probs.masked_select(mask)).sum().cpu())
            total_tokens += int(mask.sum().cpu())
            progress.advance(
                item=f"states={start}:{min(start + effective_batch_size, len(examples))}"
            )
        if total_tokens == 0:
            raise ValueError("Reference NLL received zero target tokens.")
        return total_nll / total_tokens

    @torch.inference_mode()
    def last_token_resid_pre(
        self,
        contexts: Sequence[Context],
        *,
        batch_size: int | None = None,
        progress_stage: str = "activation_cache",
    ) -> Tensor:
        """Return CPU float32 activations with shape [examples, layers, hidden]."""
        effective_batch_size = batch_size or self.config.batch_size
        chunks: list[Tensor] = []
        total = (len(contexts) + effective_batch_size - 1) // effective_batch_size
        progress = ProgressTracker(progress_stage, total)
        for start in range(0, len(contexts), effective_batch_size):
            batch_contexts = contexts[start : start + effective_batch_size]
            cache: list[Tensor | None] = [None] * self.n_layers

            def make_hook(layer: int, batch_cache: list[Tensor | None] = cache):
                def hook(_module, inputs):
                    activation = inputs[0] if isinstance(inputs, tuple) else inputs
                    batch_cache[layer] = activation[:, -1, :].detach().to("cpu", torch.float32)

                return hook

            hooks = [(block, make_hook(layer)) for layer, block in enumerate(self.blocks)]
            batch = self.tokenize(batch_contexts)
            with temporary_hooks(pre_hooks=hooks):
                self.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    use_cache=False,
                )
            if any(value is None for value in cache):
                raise RuntimeError("Failed to cache one or more transformer layers.")
            chunks.append(torch.stack([value for value in cache if value is not None], dim=1))
            progress.advance(item=f"states={start}:{start + len(batch_contexts)}")
        if not chunks:
            raise ValueError("Cannot cache activations for an empty context list.")
        return torch.cat(chunks, dim=0)

    @torch.inference_mode()
    def resid_pre_at_position(
        self,
        contexts: Sequence[Context],
        position: int,
        *,
        batch_size: int | None = None,
        progress_stage: str = "activation_cache",
    ) -> Tensor:
        """Return CPU float32 resid-pre activations as [examples, layers, hidden].

        ``position`` is a negative index into the rendered chat prompt. Using one
        semantic post-instruction position keeps the comparison aligned across
        variable-length chat and agent contexts.
        """
        if position >= 0:
            raise ValueError("Activation position must be a negative token index.")
        if not contexts:
            raise ValueError("Cannot cache activations for an empty context list.")
        effective_batch_size = batch_size or self.config.batch_size
        chunks: list[Tensor] = []
        total = (len(contexts) + effective_batch_size - 1) // effective_batch_size
        progress = ProgressTracker(
            progress_stage,
            total,
            {"states": len(contexts), "position": position},
        )
        for start in range(0, len(contexts), effective_batch_size):
            batch_contexts = contexts[start : start + effective_batch_size]
            cache: list[Tensor | None] = [None] * self.n_layers

            def make_hook(layer: int, batch_cache: list[Tensor | None] = cache):
                def hook(_module, inputs):
                    activation = inputs[0] if isinstance(inputs, tuple) else inputs
                    if activation.shape[1] < abs(position):
                        raise ValueError(
                            f"Prompt has {activation.shape[1]} tokens but position "
                            f"{position} was requested."
                        )
                    batch_cache[layer] = (
                        activation[:, position, :].detach().to("cpu", torch.float32)
                    )

                return hook

            hooks = [(block, make_hook(layer)) for layer, block in enumerate(self.blocks)]
            batch = self.tokenize(batch_contexts)
            with temporary_hooks(pre_hooks=hooks):
                self.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    use_cache=False,
                )
            if any(value is None for value in cache):
                raise RuntimeError("Failed to cache one or more transformer layers.")
            chunks.append(torch.stack([value for value in cache if value is not None], dim=1))
            progress.advance(
                item=f"states={start}:{min(start + effective_batch_size, len(contexts))}"
            )
        return torch.cat(chunks, dim=0)


def refusal_log_odds(logits: Tensor, refusal_token_ids: Tensor, epsilon: float = 1e-8) -> Tensor:
    probabilities = logits.to(torch.float64).softmax(dim=-1)
    token_ids = refusal_token_ids.to(probabilities.device)
    refusal_probability = probabilities[:, token_ids].sum(dim=-1)
    non_refusal_probability = 1.0 - refusal_probability
    return torch.log(refusal_probability + epsilon) - torch.log(non_refusal_probability + epsilon)


def forward_kl_divergence(
    reference_logits: Tensor,
    intervention_logits: Tensor,
    epsilon: float = 1e-6,
) -> Tensor:
    if reference_logits.shape != intervention_logits.shape:
        raise ValueError("KL inputs must have identical shapes.")
    reference = reference_logits.to(torch.float64)
    intervention = intervention_logits.to(torch.float64)
    reference_probs = reference.softmax(dim=-1)
    intervention_probs = intervention.softmax(dim=-1)
    return (
        reference_probs
        * (torch.log(reference_probs + epsilon) - torch.log(intervention_probs + epsilon))
    ).sum(dim=-1)
