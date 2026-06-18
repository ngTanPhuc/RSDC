#!/usr/bin/env python3
import argparse

from torch.utils.data import DataLoader

from configs.paths import *
from src.rsdc.training.train_geoai import train_geoai
import torch
import geoai.train as gtrain
from configs.geoai_hyperparams import hyperparams
import glob
from src.rsdc.evaluation.evaluate_counting import evaluate_counting_geoai
from src.rsdc.utils.geoai_utils import wandb_patch
import wandb
import os


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
        "base_model",
        type=str,
        choices=["maskrcnn, fasterrcnn"],
        help="Choose 1 of 2 base models: maskrcnn or fasterrcnn"
    )

    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=150,
        help="Number of training epochs. (Default: 150)"
    )

    # 3. Parse the arguments
    args = parser.parse_args()

    # 4. Dynamically locate the dataset YAML and output directory
    project_output_dir = EXPERIMENTS_DIR / f"{args.output_dir}"

    print(f"[INFO] Saving results to: {project_output_dir}")

    # 5. Define the base model and set up the hyperparams
    hyperparams["name"] = "RSDC-MaskRCNN" if args.base_mode == "maskrcnn" else "RSDC-FasterRCNN_resnet50"
    hyperparams["model_name"] = "maskrcnn_resnet50_fpn" if args.base_mode == "maskrcnn" else "fasterrcnn_resnet50_fpn_v2"
    hyperparams["num_epochs"] = args.epochs

    # 6. Initialize wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project=hyperparams["project"],
        name=hyperparams["name"],
        config=hyperparams
    )

    # 7. Define the wandb patch
    wandb_patch(project_output_dir)

    # 8. Train
    print("[INFO] Starting training...")
    train_geoai(project_output_dir)

    # 9. Evaluate the model with the test dataset
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model_path = project_output_dir / "models/best_model.pth"

    # Define the model architecture
    model = gtrain.get_detection_model(
        model_name=hyperparams["model_name"],
        num_classes=2,
        num_channels=hyperparams["num_channels"],
        pretrained=False
    )

    # Load the trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    print("✅ Best Model loaded successfully")

    # Create test loader
    image_paths = sorted(glob.glob(str(TEST_IMG_DIR / "*.tif")))
    label_paths = sorted(glob.glob(str(TEST_LABEL_DIR / "*.tif")))

    test_dataset = gtrain.ObjectDetectionDataset(
        image_paths=image_paths,
        label_paths=label_paths,
        transforms=gtrain.get_transform(train=False),
        num_channels=hyperparams["num_channels"]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=gtrain.collate_fn,
        num_workers=2
    )

    print(f"✅ Test loader created with {len(test_dataset)} images")

    # Evaluate the model
    basic_metrics = gtrain.evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        use_mask_iou=True if args.base_model == "maskrcnn" else False  # true for Mask R-CNN
    )

    coco_metrics = gtrain.evaluate_coco_metrics(
        model=model,
        data_loader=test_loader,
        device=device,
        class_names=["riceseedling"],
        verbose=True
    )

    print("\n" + "=" * 40)
    print("FINAL TEST RESULTS")
    print("=" * 40)
    print(f"IoU: {basic_metrics.get('IoU', 0):.4f}")
    for k, v in coco_metrics.items():
        print(f"{k:15}: {v:.4f}")

    r2, rmse, mape = evaluate_counting_geoai(model, test_loader)
    print(f"Counting -> R2: {r2:.4f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")

    # 10. Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()