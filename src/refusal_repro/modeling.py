import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

def _resolve_dtype(dtype):
    if dtype == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

def load_model_and_tokenizer(model_name, dtype="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    kwargs = {
        "torch_dtype": _resolve_dtype(dtype),
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        # This is cuda:0 within the CUDA_VISIBLE_DEVICES view, not necessarily physical GPU 0.
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer

def runtime_metadata(model=None, tokenizer=None):
    cuda_available = torch.cuda.is_available()
    visible_gpu_count = torch.cuda.device_count() if cuda_available else 0
    metadata = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": cuda_available,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_gpu_count": visible_gpu_count,
        "current_cuda_device": torch.cuda.current_device() if cuda_available else None,
        "visible_gpu_names": [
            torch.cuda.get_device_name(i) for i in range(visible_gpu_count)
        ] if cuda_available else [],
    }

    if model is not None:
        param = next(model.parameters())
        metadata["model_parameter_device"] = str(param.device)
        metadata["model_parameter_dtype"] = str(param.dtype)

    if tokenizer is not None:
        metadata["tokenizer_name_or_path"] = getattr(tokenizer, "name_or_path", None)
        metadata["tokenizer_class"] = tokenizer.__class__.__name__
        metadata["padding_side"] = tokenizer.padding_side
        metadata["truncation_side"] = tokenizer.truncation_side
        metadata["pad_token_id"] = tokenizer.pad_token_id
        metadata["eos_token_id"] = tokenizer.eos_token_id
        metadata["has_chat_template"] = bool(getattr(tokenizer, "chat_template", None))

    return metadata

def get_decoder_layers(model):
    candidates = [
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for a, b in candidates:
        obj = getattr(model, a, None)
        if obj is not None and hasattr(obj, b):
            return getattr(obj, b)
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    raise ValueError("Could not locate decoder layers for this model architecture.")
