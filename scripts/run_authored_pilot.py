"""Fixed-coordinate Qwen pilot. No tools execute; all outputs require review."""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def audit(rows):
    assert len(rows) == 192
    assert len({r['pair_id'] for r in rows}) == 192
    families = {}
    counts = Counter()
    prompts = set()
    for r in rows:
        counts[r['category'], r['split']] += 1
        families.setdefault(r['scenario_family'], set()).add(r['split'])
        t = r['task_template']
        assert t.count('{{risk_context}}') == 1
        for side in ('benign', 'harmful'):
            prompt = t.replace('{{risk_context}}', r[f'{side}_context'])
            assert '{{' not in prompt and '}}' not in prompt
            assert prompt not in prompts
            prompts.add(prompt)
        assert r['benign_context'] != r['harmful_context']
        b, h = [t.replace('{{risk_context}}', r[f'{side}_context']) for side in ('benign', 'harmful')]
        # Disable character autojunk: repeated letters in long nearly identical
        # prompts must not be discarded as common junk characters.
        assert SequenceMatcher(None, b, h, autojunk=False).ratio() >= 0.72
        sizes = [len(re.findall(r'\b\w+\b', text)) for text in (b, h)]
        assert min(sizes) >= 20 and max(sizes) / min(sizes) <= 1.2
    assert all(len(s) == 1 for s in families.values())
    for category in {r['category'] for r in rows}:
        assert [counts[category, s] for s in ('train', 'val', 'test')] == [16, 8, 8]
    return {'pairs': len(rows), 'scenario_families': len(families),
            'passed_structural': True, 'semantic_human_review_complete': False,
            'counts': {c: {s: counts[c, s] for s in ('train', 'val', 'test')}
                       for c in sorted({r['category'] for r in rows})}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['audit', 'extract', 'evaluate'])
    p.add_argument('--data', type=Path, default=Path('datasets/authored_pilot_v1/pairs.jsonl'))
    p.add_argument('--output', type=Path, default=Path('runs/authored_pilot_v1'))
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--revision', default='main')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--scales', type=float, nargs='+', default=[-1, 0, 1])
    p.add_argument('--max-new-tokens', type=int, default=192)
    args = p.parse_args()
    rows = load_rows(args.data)
    report = audit(rows)
    print(json.dumps(report, indent=2), flush=True)
    if args.stage == 'audit':
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for this unquantized pilot')
    args.output.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16,
        device_map='auto', attn_implementation='sdpa').eval()
    assert model.config.model_type == 'qwen2'
    block = model.model.layers[19]  # input to block 19 = resid_pre, zero-based
    device = model.get_input_embeddings().weight.device
    dataset_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
    fingerprint = {'dataset_sha256': dataset_sha, 'model': args.model,
                   'model_commit': model.config._commit_hash,
                   'tokenizer_sha256': hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest(),
                   'chat_template_sha256': hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                   'layer': 19, 'position': -4, 'hook': 'resid_pre',
                   'intervention': 'prefill only, single position; no decode intervention',
                   'torch': str(torch.__version__)}

    suffix_ids = None

    def encode(row, side):
        nonlocal suffix_ids
        prompt = row['task_template'].replace('{{risk_context}}', row[f'{side}_context'])
        messages = [{'role': 'system', 'content': row['system_prompt']},
                    {'role': 'user', 'content': prompt}]
        ids = tok.apply_chat_template(messages, tools=row['tools'], tokenize=True,
                                      add_generation_prompt=True)
        # Verify the semantic location, not merely that an index is in range.
        end = max(i for i, v in enumerate(ids) if v == tok.convert_tokens_to_ids('<|im_end|>'))
        assert end == len(ids) - 5, 'Chat suffix changed: -4 no longer follows user end'
        current = ids[end:]
        if suffix_ids is None:
            suffix_ids = current
        assert current == suffix_ids, 'Inconsistent post-user chat-template suffix'
        assert tok.decode([ids[-4]]) == '\n', 'Unexpected Qwen -4 token'
        return {'input_ids': torch.tensor([ids], device=device),
                'attention_mask': torch.ones((1, len(ids)), dtype=torch.long, device=device)}

    # Fail before extracting/evaluating if ANY input has the wrong anchor.
    for r in rows:
        for side in ('benign', 'harmful'):
            encode(r, side)
    fingerprint['suffix_ids'] = suffix_ids
    fingerprint['suffix_tokens'] = tok.convert_ids_to_tokens(suffix_ids)
    (args.output / 'coordinate_audit.json').write_text(json.dumps(fingerprint, indent=2))

    if args.stage == 'extract':
        deltas = {}
        pair_activations = {}
        with torch.inference_mode():
            for r in rows:
                if r['split'] != 'train':
                    continue
                cache = {}
                for side in ('benign', 'harmful'):
                    def hook(module, inputs, kwargs):
                        h = inputs[0] if inputs else kwargs['hidden_states']
                        cache[side] = h[0, -4].detach().float().cpu()
                    handle = block.register_forward_pre_hook(hook, with_kwargs=True)
                    try:
                        model(**encode(r, side), use_cache=False)
                    finally:
                        handle.remove()
                delta = cache['harmful'] - cache['benign']
                deltas.setdefault(r['category'], []).append(delta)
                pair_activations[r['pair_id']] = cache
                print('extracted', r['pair_id'], flush=True)
        means = {c: torch.stack(v).mean(0) for c, v in deltas.items()}
        means['global'] = torch.stack(list(means.values())).mean(0)
        units = {c: v / v.norm().clamp_min(1e-12) for c, v in means.items()}
        g = torch.Generator().manual_seed(42)
        random = torch.randn(means['global'].shape, generator=g)
        units['random'] = random / random.norm()
        torch.save({'fingerprint': fingerprint, 'means': means, 'units': units,
                    'alpha_reference': float(means['global'].norm()),
                    'pair_activations': pair_activations}, args.output / 'directions.pt')
        names = list(units)
        cosine = {a: {b: float(units[a] @ units[b]) for b in names} for a in names}
        (args.output / 'cosine.json').write_text(json.dumps(cosine, indent=2))
        return

    bank = torch.load(args.output / 'directions.pt', map_location='cpu', weights_only=True)
    for key in ('dataset_sha256', 'model_commit', 'tokenizer_sha256', 'chat_template_sha256'):
        assert bank['fingerprint'][key] == fingerprint[key], f'Extraction/evaluation mismatch: {key}'
    conditions = [('baseline', 0.0)] + [(name, scale) for name in bank['units']
                                       for scale in args.scales if scale != 0]
    settings = {**fingerprint, 'split': args.split, 'scales': args.scales,
                'max_new_tokens': args.max_new_tokens, 'alpha_reference': bank['alpha_reference']}
    settings_hash = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    path = args.output / f'responses_{args.split}_{settings_hash}.jsonl'
    (args.output / f'settings_{settings_hash}.json').write_text(json.dumps(settings, indent=2))
    done = {(r['pair_id'], r['side'], r['direction'], r['scale']) for r in load_rows(path)} if path.exists() else set()
    with path.open('a', encoding='utf-8') as f, torch.inference_mode():
        for r in rows:
            if r['split'] != args.split:
                continue
            for side in ('benign', 'harmful'):
                encoded = encode(r, side)
                width = encoded['input_ids'].shape[1]
                for name, scale in conditions:
                    key = (r['pair_id'], side, name, scale)
                    if key in done:
                        continue
                    handle = None
                    if scale:
                        vector = bank['units'][name]
                        def steer(module, inputs, kwargs):
                            h = inputs[0] if inputs else kwargs['hidden_states']
                            if h.shape[1] != width:
                                return  # cached decode must stay untouched
                            updated = h.clone()
                            updated[:, -4] += (scale * bank['alpha_reference'] * vector).to(h)
                            if inputs:
                                return (updated, *inputs[1:]), kwargs
                            return inputs, {**kwargs, 'hidden_states': updated}
                        handle = block.register_forward_pre_hook(steer, with_kwargs=True)
                    try:
                        seq = model.generate(**encoded, do_sample=False, use_cache=True,
                            max_new_tokens=args.max_new_tokens, pad_token_id=tok.eos_token_id)
                    finally:
                        if handle:
                            handle.remove()
                    text = tok.decode(seq[0, width:], skip_special_tokens=True)
                    tool_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.S)
                    malformed = text.count('<tool_call>') != len(tool_blocks)
                    for item in tool_blocks:
                        try:
                            obj = json.loads(item)
                            malformed |= not isinstance(obj, dict) or not {'name', 'arguments'} <= obj.keys()
                        except ValueError:
                            malformed = True
                    record = {'pair_id': r['pair_id'], 'category': r['category'],
                        'scenario_family': r['scenario_family'], 'side': side,
                        'direction': name, 'scale': scale, 'text': text,
                        'truncated': seq.shape[1] - width >= args.max_new_tokens,
                        'has_tool_call': bool(tool_blocks), 'malformed_tool_json': malformed,
                        'human_refusal': None, 'human_risky_action': None,
                        'human_task_success': None, 'human_quality': None}
                    f.write(json.dumps(record) + '\n')
                    f.flush()
                    print('generated', *key, flush=True)
    print('Review responses before claiming a refusal direction:', path, flush=True)


if __name__ == '__main__':
    main()
