#!/bin/bash
#SBATCH --job-name=qwen-negotiation
#SBATCH --output=qwen_buysell_%j.log
#SBATCH --error=qwen_buysell_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=x86  # Change to the specific GPU partition name (e.g., 'gpu' or check 'sinfo')
#SBATCH --gres=gpu:1     # Request 1 GPU

# Load the same module used during creation
module load Python/3.12.3-GCCcore-13.3.0

# Activate the x86/GPU environment created earlier
source venv_x86/bin/activate

# Set HF Token if accessing a gated model (Qwen usually requires it)

# Run the game
python runner/buysell_qwen.py