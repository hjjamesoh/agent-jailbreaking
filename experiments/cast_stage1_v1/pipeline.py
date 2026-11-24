"""Stage 1 GPU operations using the byte-verified, pinned official CAST library."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

from data import (
    ALPHAS,
    LAYERS,
    MODEL,
    MODEL_REVISION,
    ROOT,
    digest,
    final_number,
    jobs,
    read_json,
    sha_file,
    validate_assets,
    validate_prepared,
    write_json,
)


def code_digest():
    return digest({p.name: sha_file(p) for p in ROOT.glob("*.py")})


def save_direction(vector, path):
    # Upstream .save JSON-encodes this dictionary without converting NumPy
    # scalars. Convert representation only; no vector/PCA/sign values change.
    vector.explained_variances = {int(k): float(v) for k, v in vector.explained_variances.items()}
    vector.save(str(path))


def official(output):
    validate_assets(output)
    os.environ.setdefault("MPLBACKEND", "Agg")
    vendor = output / "vendor"
    sys.path.insert(0, str(vendor))
    # Keep upstream-generated log/figure directories inside this experiment.
    os.chdir(output)
    module = importlib.import_module("activation_steering")
    actual = Path(module.__file__).resolve()
    if not actual.is_relative_to(vendor.resolve()):
        raise RuntimeError(f"Wrong activation_steering module imported: {actual}")
    config = importlib.import_module("activation_steering.config")
    config.GlobalConfig.log_directory = str(output / "upstream_logs")
    config.GlobalConfig.set_verbose(False)
    return module


def load_model(output, local_only=False):
    import numpy as np
    import sklearn
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. This experiment must run on the GPU server.")
    np.random.seed(42)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=local_only)
    tok.padding_side = "left"
    # Match SteeringVector.train's upstream pad id explicitly before fingerprinting.
    tok.pad_token_id = 0
    print(f"MODEL_LOADING: {MODEL}@{MODEL_REVISION}; float16; visible cuda:0", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=MODEL_REVISION,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        local_files_only=local_only,
    ).eval()
    if model.config.model_type != "llama" or len(model.model.layers) != 32:
        raise ValueError("Expected the pinned 32-layer Hermes-2-Pro Llama-3 8B model")
    source = read_json(ROOT / "sources.json")
    fingerprint = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "resolved_commit": getattr(model.config, "_commit_hash", None),
        "upstream_revision": source["ibm_revision"],
        "sources_sha256": sha_file(ROOT / "sources.json"),
        "tokenizer_sha256": hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest(),
        "chat_template_sha256": digest(tok.chat_template),
        "model_config_sha256": digest(model.config.to_dict()),
        "dtype": "float16",
        "attention": "sdpa",
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    return model, tok, fingerprint


def download_model(output):
    from huggingface_hub import snapshot_download

    print(
        f"MODEL_DOWNLOAD: {MODEL}@{MODEL_REVISION}. GPU memory stays idle during download.",
        flush=True,
    )
    path = Path(
        snapshot_download(
            MODEL,
            revision=MODEL_REVISION,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model*.safetensors",
                "model.safetensors.index.json",
                "tokenizer*",
                "special_tokens_map.json",
            ],
        )
    )
    index = read_json(path / "model.safetensors.index.json")
    missing = [
        name
        for name in set(index["weight_map"].values())
        if not (path / name).is_file() or (path / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(f"Incomplete model cache: {missing}")
    write_json(
        output / "model_snapshot.json",
        {
            "model": MODEL,
            "revision": MODEL_REVISION,
            "path": str(path),
            "shards": len(set(index["weight_map"].values())),
        },
    )
    print(f"MODEL_DOWNLOAD_COMPLETE: {path}", flush=True)


def extract(output, batch_size=16, local_only=False):
    import numpy as np

    upstream = official(output)
    model, tok, fingerprint = load_model(output, local_only)
    alpaca = read_json(output / "vendor/docs/demo-data/alpaca.json")
    prefixes = read_json(output / "vendor/docs/demo-data/behavior_refusal.json")
    questions = [r["question"] for r in alpaca["train"][:100]]
    suffixes = list(
        zip(prefixes["non_compliant_responses"][:100], prefixes["compliant_responses"][:100])
    )
    dataset = upstream.SteeringDataset(tok, [(q, q) for q in questions], suffixes=suffixes)
    if len(dataset.formatted_dataset) != 10000:
        raise ValueError("Expected 10,000 official question/suffix combinations")
    settings = {
        "fingerprint": fingerprint,
        "batch_size": batch_size,
        "layers": LAYERS,
        "method": "pca_pairwise",
        "accumulate_last_x_tokens": "suffix-only",
        "data_sha256": digest({"questions": questions, "suffixes": suffixes}),
        "official_semantics": "Unmodified upstream pooling/PCA/sign orientation; "
        "suffix-only uses fixed length of suffixes[0][0], including for other prefixes.",
        "code_sha256": code_digest(),
    }
    state = output / "extraction"
    state.mkdir(exist_ok=True)
    manifest = state / "settings.json"
    if manifest.exists() and read_json(manifest) != settings:
        raise ValueError("Extraction settings changed. Use a fresh --output folder.")
    write_json(manifest, settings)
    target = output / "refusal.svec"
    if target.exists():
        meta = read_json(output / "direction_manifest.json")
        if meta["extraction_settings_sha256"] != digest(settings) or meta[
            "svec_sha256"
        ] != sha_file(target):
            raise ValueError("Direction manifest/checksum mismatch")
        print(f"EXTRACT_ALREADY_COMPLETE: {target}", flush=True)
        return

    # Transparent caching adapter: each batch is still processed by the original
    # official batched_get_hiddens. Pooling, padding, PCA and orientation are not rewritten.
    implementation = importlib.import_module("activation_steering.steering_vector")
    original = implementation.batched_get_hiddens

    def cached_hiddens(
        model,
        tokenizer,
        inputs,
        hidden_layer_ids,
        batch_size,
        accumulate_last_x_tokens=1,
        suffixes=None,
    ):
        chunks = defaultdict(list)
        total = (len(inputs) + batch_size - 1) // batch_size
        started, newly_processed = time.monotonic(), 0
        for offset in range(0, len(inputs), batch_size):
            strings = inputs[offset : offset + batch_size]
            key = digest(strings)[:16]
            path = state / f"batch_{offset:06d}_{key}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as values:
                    payload = {int(k): values[k] for k in values.files}
            else:
                payload = original(
                    model,
                    tokenizer,
                    strings,
                    hidden_layer_ids,
                    batch_size,
                    accumulate_last_x_tokens,
                    suffixes,
                )
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, **{str(k): v for k, v in payload.items()})
                temporary.replace(path)
                newly_processed += 1
            for layer in hidden_layer_ids:
                value = payload[layer]
                if (
                    value.shape != (len(strings), model.config.hidden_size)
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid activation checkpoint: {path}")
                chunks[layer].append(value)
            done = offset // batch_size + 1
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "stage": "extract",
                        "batch": done,
                        "total": total,
                        "eta_s": round(elapsed / newly_processed * (total - done))
                        if newly_processed
                        else None,
                    }
                ),
                flush=True,
            )
        return {layer: np.concatenate(values) for layer, values in chunks.items()}

    implementation.batched_get_hiddens = cached_hiddens
    try:
        vector = upstream.SteeringVector.train(
            model,
            tok,
            dataset,
            hidden_layer_ids=LAYERS,
            batch_size=batch_size,
            method="pca_pairwise",
            accumulate_last_x_tokens="suffix-only",
        )
    finally:
        implementation.batched_get_hiddens = original
    temporary = output / "refusal.tmp.svec"
    save_direction(vector, temporary)
    temporary.replace(target)
    write_json(
        output / "direction_manifest.json",
        {
            "fingerprint": fingerprint,
            "extraction_settings_sha256": digest(settings),
            "svec_sha256": sha_file(target),
            "layers": LAYERS,
            "paper_reference_strength": 1.7,
            "qualification": "Official-library candidate; behavior has not yet been verified.",
        },
    )
    print(f"EXTRACT_COMPLETE: {target}", flush=True)


def generate_one(wrapper, tok, prompt, vector, alpha, max_tokens):
    """Batch size one, as upstream LeashLayer edits hidden_states[0]."""
    import torch
    from activation_steering.leash_layer import LeashLayer

    wrapper.reset_leash_to_default()
    try:
        if vector is not None:
            wrapper.steer(
                behavior_vector=vector,
                behavior_layer_ids=LAYERS,
                behavior_vector_strength=alpha,
                use_ooi_preventive_normalization=False,
                apply_behavior_on_first_call=True,
            )
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        # Same formatting/tokenization route as upstream MalleableModel.respond.
        encoded = tok(formatted, return_tensors="pt").to(wrapper.device)
        width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            sequence = wrapper.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_tokens,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
                repetition_penalty=1.0,
            )[0, width:]
        eos = wrapper.model.generation_config.eos_token_id or tok.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos)
        ids = sequence.tolist()
        terminated = bool(ids and ids[-1] in eos)
        text = tok.decode(ids, skip_special_tokens=True)
        return {
            "text": text,
            "token_ids": ids,
            "generated_tokens": len(ids),
            "truncated": len(ids) >= max_tokens and not terminated,
            "empty": not text.strip(),
            "prompt_token_ids": encoded["input_ids"][0].tolist(),
            "layer_forward_calls": dict(LeashLayer.forward_calls),
        }
    finally:
        wrapper.reset_leash_to_default()


def evaluate(output, max_tokens=384, local_only=False):
    import numpy as np
    import torch

    upstream = official(output)
    prompts = validate_prepared(output)
    model, tok, fingerprint = load_model(output, local_only)
    meta = read_json(output / "direction_manifest.json")
    if (
        fingerprint != meta["fingerprint"]
        or sha_file(output / "refusal.svec") != meta["svec_sha256"]
    ):
        raise ValueError("Extraction/evaluation model or direction mismatch")
    vector = upstream.SteeringVector.load(str(output / "refusal.svec"))
    rng = np.random.default_rng(42)
    random_directions = {}
    for layer, direction in vector.directions.items():
        unit = direction / np.linalg.norm(direction)
        random_vector = rng.normal(size=unit.shape)
        random_vector -= (random_vector @ unit) * unit
        random_directions[layer] = (
            random_vector / np.linalg.norm(random_vector) * np.linalg.norm(direction)
        ).astype(np.float32)
    random_vector = upstream.SteeringVector(
        vector.model_type, random_directions, vector.explained_variances
    )
    plan = list(jobs(prompts))
    settings = {
        "fingerprint": fingerprint,
        "direction_sha256": meta["svec_sha256"],
        "prompts_sha256": digest(prompts),
        "max_tokens": max_tokens,
        "layers": LAYERS,
        "alphas": ALPHAS,
        "random_seed": 42,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "batch_size": 1,
        "code_sha256": code_digest(),
        "intervention": "official LeashLayer: pre-block input; all prefill positions "
        "and every cached decode pass; no condition gate; no OOI renormalization",
    }
    run = output / ("evaluate_" + digest(settings)[:12])
    run.mkdir(exist_ok=True)
    (run / "records").mkdir(exist_ok=True)
    write_json(run / "settings.json", settings)
    write_json(run / "plan.json", plan)
    (output / "latest_run.txt").write_text(str(run) + "\n", encoding="utf-8")
    print(f"RUN_DIR={run}", flush=True)
    lookup = {r["id"]: r for r in prompts}
    wrapper = upstream.MalleableModel(model, tok)
    started, generated = time.monotonic(), 0
    for index, job in enumerate(plan):
        destination = run / "records" / (job["key"] + ".json")
        if destination.exists():
            previous = read_json(destination)
            if previous["key"] != job["key"] or previous["prompt_id"] != job["prompt_id"]:
                raise ValueError("Corrupt result checkpoint")
            continue
        row = lookup[job["prompt_id"]]
        selected = {"none": None, "refusal": vector, "random": random_vector}[job["direction"]]
        response = generate_one(wrapper, tok, row["prompt"], selected, job["alpha"], max_tokens)
        answer = final_number(response["text"]) if row["kind"] == "benign" else None
        correct = (
            None
            if row["kind"] != "benign"
            else (answer is not None and answer == final_number("#### " + row["expected"]))
        )
        write_json(
            destination,
            {
                **job,
                **row,
                **response,
                "numeric_correct": correct,
                "math_answer_parsed": answer is not None if row["kind"] == "benign" else None,
                "strict_math_success": bool(
                    correct and not response["truncated"] and not response["empty"]
                )
                if row["kind"] == "benign"
                else None,
            },
        )
        generated += 1
        elapsed = time.monotonic() - started
        print(
            json.dumps(
                {
                    "stage": "evaluate",
                    "completed": index + 1,
                    "total": len(plan),
                    "arm": job["arm"],
                    "prompt_id": row["id"],
                    "eta_s": round(elapsed / generated * (len(plan) - index - 1)),
                    "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                }
            ),
            flush=True,
        )
    if len(list((run / "records").glob("*.json"))) != len(plan):
        raise ValueError("Evaluation incomplete")
    write_json(run / "COMPLETE.json", {"expected": len(plan), "completed": len(plan)})
    print(f"EVALUATE_COMPLETE: {run}", flush=True)
