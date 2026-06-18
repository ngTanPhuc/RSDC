import geoai
from configs.geoai_hyperparams import hyperparams
from configs.paths import *


def train_geoai(output_dir):
    geoai.train_MaskRCNN_model(
        model_name=hyperparams["model_name"],
        images_dir=f"{GEOAI_DATASET_DIR}/train/images",
        labels_dir=f"{GEOAI_DATASET_DIR}/train/labels",
        output_dir=f"{output_dir}/models",
        num_channels=hyperparams["num_channels"],
        pretrained=True,
        batch_size=hyperparams["batch_size"],
        num_epochs=hyperparams["num_epochs"],
        learning_rate=hyperparams["learning_rate"],
        val_split=hyperparams["val_split"],
    )