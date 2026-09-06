Agent Jailbreaking Refusal-Direction Experiments

This repository contains the experiment scaffold for studying whether refusal
directions detected in base LLMs can be used to modulate refusal behavior inside
an agent environment.

The previous paper-reproduction track is paused. The current plan is:

1. Experiment 0: detect refusal directions in available base LLM models.
2. Experiment 1: build an agent environment around one of those LLMs, then add
   or remove the detected refusal direction at the selected layer/position and
   compare agent behavior.

Project structure:

  experiments/
    exp0_llm_refusal_dir/
      run.py
      data/
        harmful_train.jsonl
        harmless_train.jsonl
        harmful_val.jsonl
        harmless_val.jsonl
  src/
    refusal_repro/
      analysis.py
      data.py
      directions.py
      metrics.py
      modeling.py
      selection.py
  scripts/
    check_server_env.py
    prepare_exp0_paper_splits.py
    validate_data.py
  tests/
    test_directions.py
  requirements.txt
  docs/
    CLEANUP_PLAN.md
    EXPERIMENT_PLAN.md
    EXPERIMENT0_STATUS.md
    MODEL_SELECTION.md
    GITHUB_UPLOAD.md

Experiment 0 method summary:

1. Load harmful and harmless instruction datasets.
2. Format each instruction with the model chat template.
3. Cache decoder-layer outputs at selected prompt positions.
4. Compute candidate directions with harmful mean minus harmless mean.
5. Select a candidate direction using a validation-set ablation proxy.
6. Save direction, metrics, run config, runtime metadata, and token-position audit artifacts.
7. Generate small validation completion artifacts for baseline, ablation, and activation-addition sanity checks.

Experiment 1 method summary:

1. Choose an LLM and selected refusal direction from Experiment 0.
2. Implement a minimal agent loop around the LLM, including prompts, tool calls,
   trajectory logging, and refusal/success evaluation.
3. Run the same agent tasks under three conditions: baseline, refusal-direction
   removal, and refusal-direction addition.
4. Apply the intervention at the layer/position detected in Experiment 0.
5. Compare final answers, intermediate trajectories, tool-use decisions, refusal
   rates, and task success/failure.

Important limitations:

- The included JSONL files are small pilot datasets for pipeline validation, not publication-quality datasets.
- Candidate selection currently uses a next-token refusal-prefix proxy.
- Completion refusal labels use a simple prefix heuristic, not a publication-grade evaluator.
- The ablation implementation is an inference-time decoder-layer hook, not persistent model weight orthogonalization.
- Dataset filtering by baseline refusal score is not implemented yet.
- CE-loss evaluation is not implemented yet.
- Do not interpret Experiment 0 projection or proxy metrics as causal evidence
  without Experiment 1 intervention results.

Setup:

  conda activate exp0
  pip install -r requirements.txt

The default model is:

  Qwen/Qwen3-8B-Base

You may need Hugging Face authentication and model access before running:

  hf auth login

Server preflight:

  python3 scripts/check_server_env.py
  python3 scripts/validate_data.py

Shared GPU server usage:

- Experiments are run on a shared remote GPU server.
- GPUs are numbered 0 through 3 on the server.
- Check GPU usage before every run.
- Use an idle GPU by setting CUDA_VISIBLE_DEVICES.
- Do not hard-code a physical GPU index in Python code.
- If CUDA_VISIBLE_DEVICES exposes one GPU, it will normally appear as cuda:0 inside the program.
- Tokenizer padding and truncation are set to left, so the final chat-template tokens are preserved under max-length truncation.

Recommended first pilot:

  nvidia-smi
  CUDA_VISIBLE_DEVICES=<AVAILABLE_GPU_ID> python3 experiments/exp0_llm_refusal_dir/run.py --run-name qwen3_8b_base_pilot_001 --batch-size 1 --limit-train 12 --limit-val 6

Paper-split pilot after preparing experiments/exp0_llm_refusal_dir/data/paper_splits/:

  nvidia-smi
  CUDA_VISIBLE_DEVICES=<AVAILABLE_GPU_ID> python3 experiments/exp0_llm_refusal_dir/run.py --model Qwen/Qwen3-8B-Base --run-name qwen3_8b_base_paper_pruned_001 --harmful-train experiments/exp0_llm_refusal_dir/data/paper_splits/harmful_train.jsonl --harmless-train experiments/exp0_llm_refusal_dir/data/paper_splits/harmless_train.jsonl --harmful-val experiments/exp0_llm_refusal_dir/data/paper_splits/harmful_val.jsonl --harmless-val experiments/exp0_llm_refusal_dir/data/paper_splits/harmless_val.jsonl --batch-size 1 --limit-train 64 --limit-val 32 --completion-eval-examples 16 --prune-layer-percentage 0.2

Full run after the pilot is stable:

  nvidia-smi
  CUDA_VISIBLE_DEVICES=<AVAILABLE_GPU_ID> python3 experiments/exp0_llm_refusal_dir/run.py --run-name qwen3_8b_base_full_001 --batch-size 2 --prune-layer-percentage 0.2

Layer selection notes:

- By default, Experiment 0 excludes the final 20% of decoder layers from candidate selection.
- This mirrors the original refusal-direction implementation's practice of avoiding late-layer directions.
- Pass --prune-layer-percentage 0.0 to evaluate all layers.
- Pass --candidate-layers explicitly to override automatic pruning.
- Direction removal/addition scales are recorded as --ablation-alpha and --addition-alpha.

Default outputs:

  runs/exp0_llm_refusal_dir/
    config.json
    metadata.json
    token_audit.json
    metrics.json
    best_direction.json
    candidate_directions.pt
    direction.pt
    selection_metrics.csv
    projection_by_layer.png
    benign_activation_addition_examples.jsonl
    completions/
      harmful_baseline.json
      harmful_ablation.json
      harmless_baseline.json
      harmless_activation_addition.json

Before trusting an Experiment 0 run:

1. Inspect metadata.json and confirm the expected CUDA_VISIBLE_DEVICES value.
2. Inspect token_audit.json and confirm selected positions correspond to the intended prompt tokens.
3. Inspect selection_metrics.csv and verify harmful refusal decreases without increasing harmless refusal under the proxy metric.
4. Inspect completions/ and benign_activation_addition_examples.jsonl for qualitative sanity.
5. Treat this as direction discovery only; agent-level claims require Experiment 1.

GitHub upload:

- See docs/GITHUB_UPLOAD.md for the commit and push checklist.
- Do not commit runs/, model weights, generated .pt artifacts, or secret tokens.

Token position convention:

- Negative positions are interpreted relative to each sample's final non-padding token.
- With the default --positions -1 -2 -3 -4 -5, -1 means the final non-padding token of the chat-templated prompt.
- token_audit.json records the truncated unpadded token index, token ID, decoded token text, and truncation status for sampled prompts.
