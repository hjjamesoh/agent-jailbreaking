Base LLM Selection

Recommendation:

1. Primary model: Qwen/Qwen3-8B-Base
2. Larger follow-up: Qwen/Qwen3-14B-Base
3. Cross-family comparison: meta-llama/Llama-3.1-8B
4. Lightweight/debug comparison: mistralai/Mistral-7B-v0.3

Why Qwen3-8B-Base first:

- It is a true base/pretraining-stage causal LM, which makes it suitable for
  Experiment 0 direction detection before agent alignment behavior is added.
- It uses Apache 2.0 licensing.
- It is small enough for fast iteration on a shared GPU server.
- Its model card reports 8.2B parameters, 36 decoder layers, BF16 weights, and
  32,768-token context length.
- The Qwen3 family has a direct 14B scaling path if the 8B experiment works.

Suggested Experiment 0 order:

1. Qwen/Qwen3-8B-Base
2. Qwen/Qwen3-14B-Base, if GPU memory and runtime are acceptable
3. meta-llama/Llama-3.1-8B, for comparison with a widely used Llama-family base
   model
4. mistralai/Mistral-7B-v0.3, only as a simple Apache 2.0 baseline because the
   official Mistral docs mark it as retired for new integrations

Suggested Experiment 1 model:

- Start with Qwen/Qwen3-8B-Base if Experiment 0 finds a stable direction.
- If the base model is too weak for agent behavior, run Experiment 1 with the
  paired post-trained model Qwen/Qwen3-8B as a secondary condition, but keep the
  distinction explicit because the direction was detected on the base model.

Models not recommended as the first target:

- Gemma 3 PT models: strong open-weight models, but the 4B/12B/27B variants are
  multimodal/text-image models. That adds unnecessary hook and processor
  complexity for the first refusal-direction experiment.
- Mistral-7B-v0.3: simple and Apache 2.0, but official docs mark it retired and
  suggest newer models for integrations.
- Very large MoE models: useful later, but they add routing and memory
  complexity before the core intervention pipeline is stable.

Source notes checked on 2026-09-07:

- Qwen/Qwen3-8B-Base Hugging Face model card:
  https://huggingface.co/Qwen/Qwen3-8B-Base
- Qwen/Qwen3-14B Hugging Face model card, including its base-model link:
  https://huggingface.co/Qwen/Qwen3-14B
- meta-llama/Llama-3.1-8B Hugging Face model card:
  https://huggingface.co/meta-llama/Llama-3.1-8B
- mistralai/Mistral-7B-v0.3 Hugging Face model card:
  https://huggingface.co/mistralai/Mistral-7B-v0.3
- Mistral 7B official docs:
  https://docs.mistral.ai/models/mistral-7b-0-3
- google/gemma-3-12b-pt Hugging Face model card:
  https://huggingface.co/google/gemma-3-12b-pt
