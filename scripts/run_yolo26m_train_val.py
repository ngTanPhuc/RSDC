#!/usr/bin/env python3
import argparse
from pathlib import Path
import wandb
import os

from configs.paths import EXPERIMENTS_DIR
from src.rsdc.training.train_yolo26m import train_yolo26m
from src.rsdc.evaluation.evaluate_counting import evaluate_counting_yolo
from configs.yolo_hyperparams import hyperparams


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

    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=150,
        help="Number of training epochs. (Default: 150)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO validation/counting. (Default: 0.25)"
    )

    # 3. Parse the arguments
    args = parser.parse_args()

    # 4. Dynamically locate the dataset YAML and output directory
    data_yaml_path = "configs/yolo_train_file.yaml"
    project_output_dir = EXPERIMENTS_DIR / f"{args.output_dir}"

    print(f"[INFO] Using dataset config: {data_yaml_path}")
    print(f"[INFO] Saving results to: {project_output_dir}")

    # 5. Set up the hyperparams
    hyperparams["num_epochs"] = args.epochs

    # 6. Initialize wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project=hyperparams["project"],
        name=hyperparams["name"],
        config=hyperparams
    )

    # 7. Train
    print("[INFO] Starting training...")
    model = train_yolo26m(data_yaml_path, project_output_dir)

    # 8. Evaluate the model with the test dataset
    metrics = model.val(
        data="configs/yolo_test_file.yaml",
        conf=args.conf,
        iou=0.5,
    )

    print("\n" + "=" * 40)
    print("FINAL TEST RESULTS")
    print("=" * 40)
    print("[TEST] map@50-95", metrics.box.map)  # map50-95
    print("[TEST] map@50", metrics.box.map50)  # map50
    print("[TEST] map@75", metrics.box.map75)  # map75
    print("[TEST] Per-image metrics", metrics.box.image_metrics)  # per-image metrics dictionary with precision, recall, F1, TP, FP, and FN

    print(f"[INFO] Counting uses YOLO confidence threshold: {args.conf}")
    r2, rmse, mape, counting_rows = evaluate_counting_yolo(metrics)
    print(f"Counting -> R2: {r2:.4f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")

    # 9. Finish wandb run
    wandb.finish()
    print("[INFO] W&B Run finished and synced successfully.")

if __name__ == "__main__":
    main()
