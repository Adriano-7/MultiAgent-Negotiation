#!/bin/bash
# ============================================================
# Launch one SLURM job per model for a given experiment.
#
# Usage:
#   bash slurm/launch_all.sh buysell_section_one
#   bash slurm/launch_all.sh buysell_section_one_full 5   # 5 runs (quick test)
# ============================================================

set -euo pipefail

EXPERIMENT="${1:?Usage: $0 <experiment_name> [num_runs]}"
NUM_RUNS="${2:-}"

# Models to run — edit this list to match your experiments.yaml
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    # "deepseek-ai/DeepSeek-V2-Lite-Chat"
    # "mistralai/Mistral-7B-Instruct-v0.3"
)

for MODEL in "${MODELS[@]}"; do
    SAFE_NAME=$(echo "$MODEL" | tr '/' '_')
    JOB_NAME="${EXPERIMENT}_${SAFE_NAME}"

    EXPORT_VARS="EXPERIMENT=${EXPERIMENT},MODEL=${MODEL}"
    [ -n "$NUM_RUNS" ] && EXPORT_VARS="${EXPORT_VARS},NUM_RUNS=${NUM_RUNS}"

    echo "Submitting: $JOB_NAME"
    sbatch --job-name="$JOB_NAME" --export="$EXPORT_VARS" slurm/run.sh
done

echo "All jobs submitted. Check queue with: squeue -u \$(whoami)"