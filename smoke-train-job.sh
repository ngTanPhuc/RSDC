#!/bin/bash
#SBATCH --job-name=smoke-train
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --nodes=1
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G


source .venv/bin/activate

echo "*************************"
uv pip list
python --version
echo "*************************"

echo "========================="
python -c "import torch; print('CUDA is available:', torch.cuda.is_available())"
echo "========================="

echo "Start training Mask-RCNN..."
python -m scripts.run_geoai_train_val -e 2 maskrcnn_2026_06_29_exp01 maskrcnn --score-threshold 0.5
echo "Finished training!"

echo "Start training Faster-RCNN..."
python -m scripts.run_geoai_train_val -e 2 fasterrcnn_2026_06_29_exp01 fasterrcnn --score-threshold 0.5
echo "Finished training!"

echo "Start training YOLO26m..."
yolo settings wandb=True
python -m scripts.run_yolo26m_train_val -e 2 yolo26m_2026_06_29_exp01 --conf 0.5
echo "Finished training!"

