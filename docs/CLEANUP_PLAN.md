Cleanup Plan

Current decision:

- Stop the previous paper-reproduction track.
- Keep the Experiment 0 refusal-direction detector as reusable infrastructure.
- Start Experiment 1 from a clean agent-intervention directory.

Local Git cleanup:

1. Commit the current direction-change documents and script wording as a reset
   point.
2. Keep generated files out of Git:
   - runs/
   - *.pt
   - *.safetensors
   - __pycache__/
   - .pytest_cache/
   - experiments/exp0_llm_refusal_dir/data/paper_splits/
3. Do not delete tracked source files unless they directly block Experiment 1.

GPU server cleanup:

1. Before deleting anything, list large generated outputs:
   - runs/
   - model cache copies
   - failed logs
   - temporary checkpoints
2. Move old reproduction outputs into an archive directory first:
   - runs/_archive_reproduction_2026-09-07/
3. Keep only artifacts that may help compare against the new experiments:
   - config.json
   - metadata.json
   - token_audit.json
   - metrics.json
   - selection_metrics.csv
   - best_direction.json
   - direction.pt, only if the run may be reused
4. Delete low-value generated clutter:
   - __pycache__/
   - .pytest_cache/
   - partial logs from failed runs
   - duplicate model-weight copies
   - abandoned run directories with no metrics or direction artifact

Experiment 1 clean start:

1. Create a new experiment directory:
   - experiments/exp1_agent_refusal_intervention/
2. Write new outputs under:
   - runs/exp1_agent_refusal_intervention/
3. Every agent run should record:
   - source Experiment 0 run directory
   - model checkpoint
   - layer and token position
   - direction path
   - intervention mode: baseline, add, or subtract
   - intervention scale
   - full trajectory logs
   - final refusal and task-success labels

Rule of thumb:

- Archive first, delete later.
- Delete generated clutter freely.
- Commit clean research-state changes early so GitHub reflects the new
  experiment direction.
