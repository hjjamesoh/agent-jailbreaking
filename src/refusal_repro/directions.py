import torch
from .modeling import get_decoder_layers

def format_prompts(tokenizer, prompts):
    chats = [[{"role": "user", "content": p}] for p in prompts]
    return [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        if callable(getattr(tokenizer, "apply_chat_template", None))
        and getattr(tokenizer, "chat_template", None)
        else c[0]["content"]
        for c in chats
    ]

def _format_batch(tokenizer, prompts, max_length):
    texts = format_prompts(tokenizer, prompts)
    return _tokenize_texts(tokenizer, texts, max_length)

def _tokenize_texts(tokenizer, texts, max_length):
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

def _position_indices_from_attention_mask(attention_mask, positions):
    token_counts = attention_mask.long().sum(dim=1)
    first_nonpad = attention_mask.long().argmax(dim=1)
    last_nonpad = first_nonpad + token_counts - 1

    indices = []
    for position in positions:
        if position < 0:
            unpadded_idx = token_counts + position
        else:
            unpadded_idx = torch.full_like(token_counts, int(position))
        idx = first_nonpad + unpadded_idx
        if idx.lt(0).any() or idx.ge(attention_mask.shape[1]).any():
            raise ValueError(
                f"Position {position} is outside at least one token sequence. "
                "Increase prompt length or change --positions."
            )
        if idx.gt(last_nonpad).any():
            raise ValueError(
                f"Position {position} points to padding for at least one token sequence. "
                "Increase prompt length or change --positions."
            )
        indices.append(idx)
    return torch.stack(indices, dim=1)

def token_position_audit(tokenizer, dataset_name, prompts, positions, max_length, max_examples=3):
    rows = []
    for prompt_index, prompt in enumerate(prompts[:max_examples]):
        formatted_text = format_prompts(tokenizer, [prompt])[0]
        full_ids = tokenizer(
            formatted_text,
            add_special_tokens=False,
            truncation=False,
        ).input_ids
        used_ids = tokenizer(
            formatted_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        ).input_ids

        token_rows = []
        for position in positions:
            absolute_index = len(used_ids) + position if position < 0 else position
            in_range = 0 <= absolute_index < len(used_ids)
            token_id = int(used_ids[absolute_index]) if in_range else None
            token_rows.append({
                "relative_position": int(position),
                "index_in_truncated_unpadded_sequence": int(absolute_index),
                "in_range": bool(in_range),
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]) if token_id is not None else None,
            })

        rows.append({
            "dataset": dataset_name,
            "prompt_index": prompt_index,
            "raw_prompt": prompt,
            "formatted_text": formatted_text,
            "num_tokens_before_truncation": len(full_ids),
            "num_tokens_after_truncation": len(used_ids),
            "truncated": len(full_ids) > len(used_ids),
            "tokenizer_truncation_side": tokenizer.truncation_side,
            "tokenizer_padding_side": tokenizer.padding_side,
            "positions": token_rows,
        })
    return rows

@torch.no_grad()
def mean_residuals(model, tokenizer, prompts, positions=(-1,), batch_size=4, max_length=1024):
    texts = format_prompts(tokenizer, prompts)
    return mean_residuals_from_texts(
        model,
        tokenizer,
        texts,
        positions=positions,
        batch_size=batch_size,
        max_length=max_length,
    )

@torch.no_grad()
def mean_residuals_from_texts(model, tokenizer, texts, positions=(-1,), batch_size=4, max_length=1024):
    layers = get_decoder_layers(model)
    device = next(model.parameters()).device
    sums = None
    count = 0

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        toks = _tokenize_texts(tokenizer, batch_texts, max_length)
        toks = {k: v.to(device) for k, v in toks.items()}
        position_indices = _position_indices_from_attention_mask(
            toks["attention_mask"],
            positions,
        )
        captured = [None] * len(layers)
        hooks = []

        for li, layer in enumerate(layers):
            def make_hook(idx):
                def hook(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captured[idx] = hidden.detach()
                return hook
            hooks.append(layer.register_forward_hook(make_hook(li)))

        try:
            model(**toks, use_cache=False)
        finally:
            for h in hooks:
                h.remove()

        batch_vals = []
        for hidden in captured:
            vals = []
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
            for p_idx in range(len(positions)):
                token_indices = position_indices[:, p_idx].to(hidden.device)
                vals.append(hidden[batch_indices, token_indices, :].float().cpu())
            batch_vals.append(torch.stack(vals, dim=1))

        stacked = torch.stack(batch_vals, dim=0)  # [L, B, P, D]
        cur_sum = stacked.sum(dim=1)               # [L, P, D]
        sums = cur_sum if sums is None else sums + cur_sum
        count += len(batch_texts)

    return sums / count

def compute_candidate_directions(harmful_means, harmless_means, eps=1e-8):
    diff = harmful_means - harmless_means
    return diff / diff.norm(dim=-1, keepdim=True).clamp_min(eps)
