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
- Apply the detected refusal direction inside the agent's model calls.
- Compare behavior when the direction is removed, added, or left untouched.

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

Suggested metrics:

- Final refusal rate.
- Task success rate.
- Tool-call count and tool-call type distribution.
- Trajectory length.
- Unsafe or policy-violating completion rate, if a suitable evaluator is added.
- Qualitative trajectory differences between baseline, removal, and addition.

Handoff from Experiment 0 to Experiment 1:

1. Select an Experiment 0 run directory.
2. Load direction.pt and best_direction.json.
3. Verify that the model checkpoint, layer, and position match the intended
   agent model.
4. Run a small agent smoke test across all three conditions.
5. Only then scale to larger agent-task batches.
