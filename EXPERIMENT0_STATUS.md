Experiment 0 Status

Target paper:

- "Refusal in Language Models Is Mediated by a Single Direction"
- Reference implementation: https://github.com/andyrdt/refusal_direction

Current implementation scope:

- Loads harmful and harmless prompt JSONL files.
- Applies the model chat template when available.
- Captures decoder-layer outputs with forward hooks.
- Computes layer/position candidate directions as mean(harmful) - mean(harmless).
- Selects a candidate direction using a validation-set next-token refusal-prefix proxy under inference-time ablation.
- Saves run config, runtime metadata, token-position audit, selection metrics, selected direction, and a projection plot.
- Generates small validation completion artifacts for baseline, ablation, and activation-addition sanity checks.

Reproducibility safeguards already implemented:

- No physical GPU index is hard-coded in run commands.
- CUDA_VISIBLE_DEVICES is recorded in metadata.json.
- Visible CUDA devices and names are recorded in metadata.json.
- Tokenizer padding and truncation sides are recorded.
- Negative positions are resolved against each sample's final non-padding token.
- token_audit.json records sampled token IDs/texts for the selected positions.
- runs/ and model artifacts are ignored by git.

Known limitations before claiming paper-level reproduction:

- The bundled datasets are tiny pilot datasets, not the original full training/evaluation datasets.
- Candidate selection uses a refusal-prefix probability proxy.
- Completion refusal labels use a simple prefix heuristic rather than a publication-grade evaluator.
- The implementation uses inference-time hooks, not persistent weight orthogonalization.
- Baseline refusal-score dataset filtering is not implemented.
- CE-loss evaluation is not implemented.
- The exact original model wrapper/tokenization utilities are not vendored.

Recommended next evidence on the GPU server:

1. Run scripts/check_server_env.py after activating the conda environment.
2. Run scripts/validate_data.py to confirm the prompt JSONL files are valid.
3. Run the README pilot command on an idle GPU.
4. Inspect metadata.json to confirm the intended CUDA_VISIBLE_DEVICES value.
5. Inspect token_audit.json to confirm the selected token positions.
6. Inspect selection_metrics.csv and metrics.json for meaningful harmful refusal reduction.
7. Inspect completions/ and benign_activation_addition_examples.jsonl for qualitative sanity.

Do not proceed to agent trajectory instrumentation until the Experiment 0 pilot behavior is stable.
