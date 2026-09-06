Experiment Plan

The previous paper-reproduction track is paused. The current research flow has
two experiments.

Experiment 0: base LLM refusal-direction detection

Goal:

- Detect refusal directions in available base LLM models.
- Identify the layer and prompt position where the direction is most useful for
  intervention.
- Produce reusable artifacts for downstream agent experiments.

Inputs:

- A target Hugging Face causal LM.
- Harmful and harmless instruction datasets.
- Candidate decoder layers and prompt positions.

Procedure:

1. Format harmful and harmless prompts with the model's chat template when
   available.
2. Capture decoder-layer activations at selected token positions.
3. Compute candidate directions as mean(harmful) - mean(harmless).
4. Select the best candidate using validation-set refusal reduction under
   inference-time direction removal.
5. Sanity-check causal relevance by removing the direction on harmful prompts
   and adding it on harmless prompts.
6. Save the selected direction, model metadata, layer, token position, metrics,
   and qualitative completions.

Primary artifacts:

- runs/exp0_llm_refusal_dir/<run>/direction.pt
- runs/exp0_llm_refusal_dir/<run>/best_direction.json
- runs/exp0_llm_refusal_dir/<run>/metadata.json
- runs/exp0_llm_refusal_dir/<run>/token_audit.json
- runs/exp0_llm_refusal_dir/<run>/metrics.json

Experiment 1: agent-environment intervention

Goal:

- Build an agent environment using the LLM selected in Experiment 0.
- Check whether harmful/harmless agent contexts produce a similar refusal direction.
- Apply the detected refusal direction inside the agent's model calls.
- Compare behavior when the direction is removed, added, or left untouched.

Current implementation:

- Entrypoint: experiments/exp1_agent_refusal_dir/run.py
- Agent utilities: src/refusal_repro/agent_env.py
- Environment: a minimal ReAct-style loop with safe mock tools.
- Modes:
  - detect: compute an agent-context refusal direction from harmful/harmless agent prompts.
  - apply: load an Experiment 0 direction.pt and apply it inside agent generation.
  - both: run both detect and apply in one job.

How this differs from Experiment 0:

- Experiment 0 formats each example as a single user-to-assistant prompt.
- Experiment 1 formats each example as an agent task with system instructions,
  tool descriptions, and an agent scratchpad.
- Experiment 0 logs completions; Experiment 1 logs trajectories, including model
  output, parsed tool call, observation, final answer, intervention mode, layer,
  position, direction path, and alpha.
- The current Experiment 1 code additionally records first_action, used_tools,
  used_policy_tool, completed_with_finish, and a heuristic final_label.

Experimental conditions:

- Baseline: no activation intervention.
- Direction removal: subtract the Experiment 0 refusal direction at the selected
  layer/position.
- Direction addition: add the Experiment 0 refusal direction at the selected
  layer/position.

Agent requirements:

- Use the same model checkpoint as the selected Experiment 0 run.
- Log prompts, model outputs, tool calls, observations, final answers, and
  intervention configuration for every trajectory.
- Keep intervention code explicit about layer index, token position, direction
  source path, alpha/scale, and whether the operation is add or subtract.
- Evaluate both final refusal behavior and intermediate agent behavior.

Suggested first smoke command:

  CUDA_VISIBLE_DEVICES=<AVAILABLE_GPU_ID> python3 experiments/exp1_agent_refusal_dir/run.py --mode both --model Qwen/Qwen3-8B-Base --run-name qwen3_8b_base_agent_smoke_001 --exp0-direction runs/exp0_llm_refusal_dir/qwen3_8b_base_paper_mid_layers_fulltrain_001/direction.pt --harmful-train experiments/exp0_llm_refusal_dir/data/paper_splits/harmful_train.jsonl --harmless-train experiments/exp0_llm_refusal_dir/data/paper_splits/harmless_train.jsonl --harmful-val experiments/exp0_llm_refusal_dir/data/paper_splits/harmful_val.jsonl --harmless-val experiments/exp0_llm_refusal_dir/data/paper_splits/harmless_val.jsonl --batch-size 1 --limit-train 64 --limit-val 16 --agent-eval-examples 8 --candidate-layers 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24

Suggested metrics:

- Final refusal rate.
- Final response label distribution:
  refusal, safe_alternative, policy_reasoning, harmful_compliance,
  unfinished_tool_call, other.
- Task success rate.
- Tool-call count and tool-call type distribution.
- Policy-tool use rate.
- Completed-with-finish rate.
- Trajectory length.
- Unsafe or policy-violating completion rate, if a suitable evaluator is added.
- Qualitative trajectory differences between baseline, removal, and addition.

Current code change notes:

- Previous Experiment 1 code treated only refusal-prefix matches as refusals.
- Current code separates safe alternatives and policy reasoning from simple
  refusal-prefix matches.
- Current code labels max-step trajectories that end in another tool call as
  unfinished_tool_call instead of treating the final tool-call text as a final answer.
- Current code uses a force-finish context after a tool observation and rejects
  non-finish tool calls during that finalization step.
- Previous code kept full generated text even when the model emitted multiple
  Action blocks in one step.
- Current code truncates each generated step after the first Action Input line,
  making one model generation correspond to one agent step.

Handoff from Experiment 0 to Experiment 1:

1. Select an Experiment 0 run directory.
2. Load direction.pt and best_direction.json.
3. Verify that the model checkpoint, layer, and position match the intended
   agent model.
4. Run a small agent smoke test across all three conditions.
5. Only then scale to larger agent-task batches.
