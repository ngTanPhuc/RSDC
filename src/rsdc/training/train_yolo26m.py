from ultralytics import YOLO
import wandb
import os
from configs.yolo_hyperparams import hyperparams
from src.rsdc.utils.yolo_utils import save_checkpoint_to_wandb


def train_yolo26m(data_yaml_file: str, output_dir: str):
    # ===============================
    # TRAINING
    # ===============================
    model = YOLO("yolo26m.pt")  # load the pretrained
    model.info()  # print the model info

    # Set up the function to save the ckpt
    model.add_callback("on_fit_epoch_end", save_checkpoint_to_wandb)

    model.train(
        data=data_yaml_file,
        epochs=hyperparams["num_epochs"],
        imgsz=hyperparams["tile_size"],
        batch=hyperparams["batch_size"],
        workers=4,
        patience=20,
        lr0=hyperparams["learning_rate"],
        lrf=1e-2,
        scale=0.3,
        fliplr=0.35,
        flipud=0.4,
        hsv_s=0.4,
        hsv_v=0.2,
        mosaic=0.8,
        mixup=0.10,
        close_mosaic=10,
        shear=5,
        name=hyperparams["name"],
        project=output_dir,
        device=0,
        save=True,
        save_period=-1
        )

    return model
