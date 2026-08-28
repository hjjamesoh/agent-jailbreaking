Experiment 0: Refusal Direction Reproduction

This directory contains a minimal reproduction scaffold for Experiment 0 of
"Refusal in Language Models Is Mediated by a Single Direction".

The immediate goal is to validate the LLM-only refusal-direction pipeline before
moving on to agent trajectory or tool-use experiments.

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
    validate_data.py
  tests/
    test_directions.py
  requirements.txt
  docs/
    EXPERIMENT0_STATUS.md
    GITHUB_UPLOAD.md

Method summary:

1. Load harmful and harmless instruction datasets.
2. Format each instruction with the model chat template.
3. Cache decoder-layer outputs at selected prompt positions.
4. Compute candidate directions with harmful mean minus harmless mean.
5. Select a candidate direction using a validation-set ablation proxy.
6. Save direction, metrics, run config, runtime metadata, and token-position audit artifacts.
7. Generate small validation completion artifacts for baseline, ablation, and activation-addition sanity checks.

Important limitations:

- The included JSONL files are small pilot datasets for pipeline validation, not publication-quality datasets.
- Candidate selection currently uses a next-token refusal-prefix proxy.
- Completion refusal labels use a simple prefix heuristic, not a publication-grade evaluator.
- The ablation implementation is an inference-time decoder-layer hook, not persistent model weight orthogonalization.
- Dataset filtering by baseline refusal score is not implemented yet.
- CE-loss evaluation is not implemented yet.
- Do not interpret projection or proxy metrics as causal evidence without intervention results.

Setup:

  conda activate exp0
  pip install -r requirements.txt

The default model is:

  meta-llama/Meta-Llama-3-8B-Instruct

You may need Hugging Face authentication and model access before running:

  huggingface-cli login

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
  CUDA_VISIBLE_DEVICES=<AVAILABLE_GPU_ID> python3 experiments/exp0_llm_refusal_dir/run.py --run-name pilot_001 --candidate-layers 8 12 16 20 24 --batch-size 1 --limit-train 8 --limit-val 4

Full run after the pilot is stable:

  nvidia-smi
  CUDA_VISIBLE_DEVICES=<AVAILABLE_GPU_ID> python3 experiments/exp0_llm_refusal_dir/run.py --run-name full_001 --batch-size 2

Default outputs:

  runs/exp0_llm_refusal_dir/
    config.json
    metadata.json
    token_audit.json
    metrics.json
    candidate_directions.pt
    direction.pt
    best_direction.json
    selection_metrics.csv
    projection_by_layer.png
    benign_activation_addition_examples.jsonl
    completions/
      harmful_baseline.json
      harmful_ablation.json
      harmless_baseline.json
      harmless_activation_addition.json

Before trusting a run:

1. Inspect metadata.json and confirm the expected CUDA_VISIBLE_DEVICES value.
2. Inspect token_audit.json and confirm selected positions correspond to the intended prompt tokens.
3. Inspect selection_metrics.csv and verify harmful refusal decreases without increasing harmless refusal under the proxy metric.
4. Inspect completions/ and benign_activation_addition_examples.jsonl for qualitative sanity.
5. Treat this as Experiment 0 validation only; do not proceed to trajectory experiments until the reproduction behavior is stable.

GitHub upload:

- See docs/GITHUB_UPLOAD.md for the commit and push checklist.
- Do not commit runs/, model weights, generated .pt artifacts, or secret tokens.

Token position convention:

- Negative positions are interpreted relative to each sample's final non-padding token.
- With the default --positions -1 -2 -3 -4 -5, -1 means the final non-padding token of the chat-templated prompt.
- token_audit.json records the truncated unpadded token index, token ID, decoded token text, and truncation status for sampled prompts.
