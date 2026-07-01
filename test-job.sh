#!/bin/bash
#SBATCH --job-name=test_python_ver
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --nodes=1
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=4G


source .venv/bin/activate

echo "Python version"
python --version

