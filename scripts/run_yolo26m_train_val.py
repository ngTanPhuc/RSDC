#!/usr/bin/env python3
import argparse
from pathlib import Path
import wandb

from configs.paths import EXPERIMENTS_DIR
from src.rsdc.training.train_yolo26m import train_yolo26m
from src.rsdc.evaluation.evaluate_counting import evaluate_counting_yolo


def main():
    # 1. Initialize the parser
    parser = argparse.ArgumentParser(
        description="A script to convert raw dataset to YOLO ready dataset"
    )

    # 2. Add require arguments (no dashes, no default value)
    parser.add_argument(
        "output_dir",
        type=str,
        help="Output directory for this training run"
    )

    # 3. Parse the arguments
    args = parser.parse_args()

    # 4. Dynamically locate the dataset YAML and output directory
    data_yaml_path = str(Path("../configs/yolo_train_file.yaml"))
    project_output_dir = EXPERIMENTS_DIR / f"{args.output_dir}"

    print(f"[INFO] Using dataset config: {data_yaml_path}")
    print(f"[INFO] Saving results to: {project_output_dir}")

    # 5. Initialize Model
    print("[INFO] Loading YOLO model...")

    # 6. Train
    print("[INFO] Starting training...")
    model = train_yolo26m(data_yaml_path)

    # 7. Evaluate the model with the test dataset
    metrics = model.val(data="/kaggle/working/datasets/test_data.yaml")
    print("\n" + "=" * 40)
    print("FINAL TEST RESULTS")
    print("=" * 40)
    print("[TEST] map@50-95", metrics.box.map)  # map50-95
    print("[TEST] map@50", metrics.box.map50)  # map50
    print("[TEST] map@75", metrics.box.map75)  # map75
    print("[TEST] Per-image metrics", metrics.box.image_metrics)  # per-image metrics dictionary with precision, recall, F1, TP, FP, and FN
    r2, rmse, mape = evaluate_counting_yolo(metrics)
    print(f"Counting -> R²: {r2:.4f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")

    # 8. Finish wandb run
    wandb.finish()
    print("[INFO] W&B Run finished and synced successfully.")

if __name__ == "__main__":
    main()