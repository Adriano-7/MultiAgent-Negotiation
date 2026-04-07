#!/bin/bash
#SBATCH --job-name=negotiation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/slurm/%x_%j.log
#SBATCH --error=logs/slurm/%x_%j.err

# ============================================================
# Generic SLURM batch script for NegotiationArena.
#
# Server-specific settings (partition, GPUs, time, etc.) are
# injected as sbatch CLI flags by launch.sh. This script reads
# environment variables for paths and modules.
#
# Required env vars (set by server profile, passed via --export):
#   CONDA_INIT_SCRIPT  — shell command to initialise conda
#   CONDA_ENV_PATH     — path to the conda environment
#   HF_HOME            — HuggingFace model cache directory
#
# Optional env vars (set by server profile):
#   MODULE_LOADS        — space-separated module names to load
#   EXTRA_ENV_VARS      — space-separated KEY=VALUE pairs to export
#
# Required env vars (set by launcher):
#   EXPERIMENT          — experiment name from configs/experiments.yaml
#
# Optional env vars (set by launcher):
#   CONFIG, MODEL, NUM_RUNS, SIZE
#
# Usage (via launch.sh — preferred):
#   SERVER=mia bash slurm/launch.sh
#
# Usage (direct — advanced):
#   source slurm/servers/mia.sh
#   sbatch --partition=$SLURM_PARTITION --time=$SLURM_TIME \
#     $SLURM_GPU_DIRECTIVE --cpus-per-task=$SLURM_CPUS_PER_TASK \
#     --export=ALL,EXPERIMENT=buysell_section_one,SIZE=very_small,\
#     CONDA_INIT_SCRIPT="$CONDA_INIT_SCRIPT",CONDA_ENV_PATH="$CONDA_ENV_PATH",\
#     HF_HOME="$HF_HOME" slurm/run.sh
# ============================================================

set -euo pipefail

# ── Server-specific module loads ─────────────────────────────
if [ -n "${MODULE_LOADS:-}" ]; then
    for mod in $MODULE_LOADS; do
        module load "$mod"
    done
fi

# ── Conda activation ────────────────────────────────────────
eval "${CONDA_INIT_SCRIPT:?ERROR: CONDA_INIT_SCRIPT not set}"
conda activate "${CONDA_ENV_PATH:?ERROR: CONDA_ENV_PATH not set}"

# ── Extra environment (e.g., offline mode) ───────────────────
if [ -n "${EXTRA_ENV_VARS:-}" ]; then
    for var in $EXTRA_ENV_VARS; do
        export "$var"
    done
fi

# ── Experiment parameters ────────────────────────────────────
EXPERIMENT="${EXPERIMENT:?ERROR: set EXPERIMENT via --export}"
CONFIG="${CONFIG:-configs/experiments.yaml}"
MODEL="${MODEL:-}"
NUM_RUNS="${NUM_RUNS:-}"
SIZE="${SIZE:-}"

# ── Environment from .env ────────────────────────────────────
set -a
source .env 2>/dev/null || true
set +a

export HF_HOME="${HF_HOME:?ERROR: HF_HOME not set}"
mkdir -p logs/slurm

# ── Build and run command ────────────────────────────────────
CMD="python runner/run_experiment.py --config ${CONFIG} --experiment ${EXPERIMENT}"
[ -n "$MODEL" ]    && CMD="$CMD --model \"$MODEL\""
[ -n "$NUM_RUNS" ] && CMD="$CMD --num_runs $NUM_RUNS"
[ -n "$SIZE" ]     && CMD="$CMD --model_group $SIZE"

echo "============================================"
echo "SERVER     : ${SERVER:-unknown}"
echo "EXPERIMENT : $EXPERIMENT"
echo "SIZE GROUP : ${SIZE:-<fallback to config default>}"
echo "MODEL      : ${MODEL:-<all from group>}"
echo "NUM_RUNS   : ${NUM_RUNS:-<from config>}"
echo "CMD        : $CMD"
echo "============================================"

eval $CMD
