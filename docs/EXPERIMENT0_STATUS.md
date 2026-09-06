Experiment 0 Status

Current role:

- Experiment 0 detects refusal directions from available base LLM models.
- Its output is a selected layer, prompt position, and refusal-direction vector
  that can be reused by Experiment 1 in an agent environment.
- The previous paper-reproduction goal is paused.

Current implementation scope:

- Experiment entrypoint: experiments/exp0_llm_refusal_dir/run.py
- Shared implementation package: src/refusal_repro/
- Pilot data: experiments/exp0_llm_refusal_dir/data/
- Optional legacy paper split converter: scripts/prepare_exp0_paper_splits.py
- Loads harmful and harmless prompt JSONL files.
- Applies the model chat template when available.
- Captures decoder-layer outputs with forward hooks.
- Computes layer/position candidate directions as mean(harmful) - mean(harmless).
- Selects a candidate direction using a validation-set next-token refusal-prefix proxy under inference-time ablation.
- Excludes the final 20% of decoder layers from automatic candidate selection
  by default; explicit --candidate-layers overrides this.
- Records removal/addition strengths with --ablation-alpha and --addition-alpha.
- Saves run config, runtime metadata, token-position audit, selection metrics, selected direction, and a projection plot.
- Generates small validation completion artifacts for baseline, ablation, and activation-addition sanity checks.

Reproducibility safeguards already implemented:

- No physical GPU index is hard-coded in run commands.
- CUDA_VISIBLE_DEVICES is recorded in metadata.json.
- Visible CUDA devices and names are recorded in metadata.json.
- Tokenizer padding and truncation sides are recorded.
- Negative positions are resolved against each sample's final non-padding token.
- token_audit.json records sampled token IDs/texts for the selected positions.
- config.json and metadata.json record the effective candidate layer list and intervention alpha values.
- runs/ and model artifacts are ignored by git.
- Converted paper_splits are ignored by git because they are derived from the original repository.

Known limitations before using this as agent evidence:

- The bundled datasets are tiny pilot datasets; replace or expand them before larger direction-discovery runs.
- Candidate selection uses a refusal-prefix probability proxy.
- Completion refusal labels use a simple prefix heuristic rather than a publication-grade evaluator.
- The implementation uses inference-time hooks, not persistent weight orthogonalization.
- Baseline refusal-score dataset filtering is not implemented.
- CE-loss evaluation is not implemented.
- Agent-level intervention is not implemented in Experiment 0.

Recommended next evidence on the GPU server:

1. Run scripts/check_server_env.py after activating the conda environment.
2. Run scripts/validate_data.py to confirm the prompt JSONL files are valid.
3. Run the README pilot command on an idle GPU.
4. Inspect metadata.json to confirm the intended CUDA_VISIBLE_DEVICES value.
5. Inspect token_audit.json to confirm the selected token positions.
6. Inspect selection_metrics.csv and metrics.json for meaningful harmful refusal reduction.
7. Inspect completions/ and benign_activation_addition_examples.jsonl for qualitative sanity.
8. Promote a run to Experiment 1 only when direction.pt, best_direction.json,
   and token_audit.json agree on the intended model/layer/position.

Experiment 1 handoff criteria:

- A concrete model checkpoint is selected.
- best_direction.json records a clear layer and position.
- direction.pt contains the selected vector and model metadata.
- Baseline, removal, and addition smoke completions show direction-sensitive
  behavior worth testing in an agent loop.
