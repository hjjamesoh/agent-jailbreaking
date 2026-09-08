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
    exp1_agent_refusal_dir/
      run.py
  src/
    refusal_repro/
      agent_env.py
      analysis.py
      data.py
      directions.py
      metrics.py
      modeling.py
      selection.py
  scripts/
    check_server_env.py
    compare_directions.py
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
2. Build a minimal ReAct-style agent loop around the LLM with safe mock tools.
3. Detect a new agent-context direction from harmful/harmless agent prompts.
4. Run the same agent tasks under three conditions: baseline, refusal-direction
   removal, and refusal-direction addition.
5. Apply the Experiment 0 direction at the selected layer/position during agent generation.
6. Compare final answers, intermediate trajectories, tool-use decisions, refusal
   rates, response labels, and task success/failure.
7. Select agent-context directions with an agent-behavior score by default:
   harmful safety-behavior reduction under ablation, minus harmless success loss
   and invalid-action increase.

Important limitations:

- The included JSONL files are small pilot datasets for pipeline validation, not publication-quality datasets.
- Experiment 0 candidate selection currently uses a next-token refusal-prefix proxy.
- Experiment 1 candidate selection uses an agent final-behavior score by default,
  with the next-token refusal-prefix proxy still available as --selection-metric next_token.
- Completion refusal labels use a simple prefix heuristic, not a publication-grade evaluator.
- The ablation implementation is an inference-time decoder-layer hook, not persistent model weight orthogonalization.
- Dataset filtering by baseline refusal score is not implemented yet.
- CE-loss evaluation is not implemented yet.
- Experiment 1 currently uses a minimal mock-tool agent environment, not a full
  browser, shell, or web-connected autonomous agent.
- Experiment 1 labels are heuristic and intended for smoke testing, not final
  safety evaluation.
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

Experiment 1 smoke test with the selected Experiment 0 direction:

  CUDA_VISIBLE_DEVICES=<AVAILABLE_GPU_ID> python3 experiments/exp1_agent_refusal_dir/run.py --mode both --model Qwen/Qwen3-8B-Base --run-name qwen3_8b_base_agent_weak_behavior_001 --agent-prompt-profile weak --selection-metric agent_behavior --behavior-selection-examples 8 --exp0-direction runs/exp0_llm_refusal_dir/qwen3_8b_base_paper_mid_layers_fulltrain_001/direction.pt --harmful-train experiments/exp0_llm_refusal_dir/data/paper_splits/harmful_train.jsonl --harmless-train experiments/exp0_llm_refusal_dir/data/paper_splits/harmless_train.jsonl --harmful-val experiments/exp0_llm_refusal_dir/data/paper_splits/harmful_val.jsonl --harmless-val experiments/exp0_llm_refusal_dir/data/paper_splits/harmless_val.jsonl --batch-size 1 --limit-train 64 --limit-val 16 --agent-eval-examples 8 --candidate-layers 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24

Repeat the same command with --agent-prompt-profile strong to compare weak and
strong safety scaffolding.

Experiment 1 outputs:

  runs/exp1_agent_refusal_dir/
    config.json
    metadata.json
    metrics.json
    agent_token_audit.json
    agent_candidate_directions.pt
    agent_direction.pt
    agent_best_direction.json
    agent_behavior_selection_metrics.csv
    agent_selection_metrics.csv              # only when --selection-metric next_token
    agent_selection_runs/
    agent_runs/
      harmful_baseline.jsonl
      harmful_subtract.jsonl
      harmful_add.jsonl
      harmless_baseline.jsonl
      harmless_subtract.jsonl
      harmless_add.jsonl

Direction similarity analysis:

  python3 scripts/compare_directions.py --run-name qwen3_exp0_vs_exp1_smoke_005 --reference-direction runs/exp0_llm_refusal_dir/qwen3_8b_base_paper_mid_layers_fulltrain_001/direction.pt --target-direction runs/exp1_agent_refusal_dir/qwen3_8b_base_agent_smoke_005/agent_direction.pt --target-candidates runs/exp1_agent_refusal_dir/qwen3_8b_base_agent_smoke_005/agent_candidate_directions.pt

This analysis does not run the model. It loads saved direction artifacts and
reports selected-direction cosine similarity plus the target candidate
layer/position cells most similar to the reference direction.

Direction similarity outputs:

  runs/direction_comparisons/
    summary.json
    candidate_cosine_top_matches.csv
    candidate_cosine_grid.pt
    candidate_cosine_heatmap.png

Difference between Experiment 0 and Experiment 1:

- Experiment 0 detects directions at a single-turn LLM generation boundary.
- Experiment 1 detects and applies directions inside an agent prompt that includes
  system instructions, tool descriptions, a task, and an agent scratchpad.
- Experiment 1 supports --agent-prompt-profile weak and strong so prompt safety
  scaffolding is an explicit experimental condition.
- Experiment 1 supports --selection-metric agent_behavior and next_token. The
  behavior score is the default because it is closer to final agent refusal,
  safe-alternative, harmful-compliance, and benign-task behavior.
- Experiment 1 logs trajectory fields: prompt context, model output, parsed tool
  call, observation, final answer, intervention mode, layer, position, direction
  path, and alpha.

Recent Experiment 1 instrumentation changes:

- Earlier code only counted looks_like_refusal from a simple refusal-prefix heuristic.
- Current code also records final_label values: refusal, safe_alternative,
  policy_reasoning, harmful_compliance, unfinished_tool_call, or other.
- Current code records first_action, used_tools, and used_policy_tool.
- Current code clips each model step to the first Action / Action Input block so
  one generated step does not accidentally contain multiple unexecuted actions.
- After a tool observation, the next agent step is marked as force_finish and
  only the finish action is accepted.
- If the model still emits a non-finish action during force_finish, the run logs
  it as an invalid finalization action and writes a deterministic fallback final answer.
- Summaries include completed_with_finish_rate, completed_with_fallback_rate,
  and invalid finalization action counts.
- lookup_fact is available for benign factual requests, while lookup_policy is
  reserved for safety-sensitive requests.
- Safe-alternative markers include recommendation-style safe redirections such
  as protecting sensitive information or focusing on ethical treatment.
- Agent-context direction detection now selects candidates by final agent
  behavior by default. The score is harmful_safety_delta minus harmless success
  loss and invalid-action increase.

GitHub upload:

- See docs/GITHUB_UPLOAD.md for the commit and push checklist.
- Do not commit runs/, model weights, generated .pt artifacts, or secret tokens.

Token position convention:

- Negative positions are interpreted relative to each sample's final non-padding token.
- With the default --positions -1 -2 -3 -4 -5, -1 means the final non-padding token of the chat-templated prompt.
- token_audit.json records the truncated unpadded token index, token ID, decoded token text, and truncation status for sampled prompts.
