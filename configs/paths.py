from pathlib import Path

# 1. Dynamically find the project root
# __file__ is the absolute path to this specific paths.py file
# .resolve() resolves any symlinks
# .parents[1] goes up two levels (from configs/paths.py -> configs/ -> RSDC/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 2. Define standard project directories
# Experiment dirs
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Root and preprocessed data dirs
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATION_DIR = RAW_DIR / "annotations"
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
RASTER_DIR = RAW_DIR / "rasters"
RASTER_DIR.mkdir(parents=True, exist_ok=True)

# Processed dirs
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# YOLO
YOLO_DATASET_DIR = PROCESSED_DIR / "yolo_dataset"
YOLO_DATASET_DIR.mkdir(parents=True, exist_ok=True)

IMG_DIR = YOLO_DATASET_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
(IMG_DIR / "train").mkdir(parents=True, exist_ok=True)
(IMG_DIR / "val").mkdir(parents=True, exist_ok=True)
(IMG_DIR / "test").mkdir(parents=True, exist_ok=True)

LABELS_DIR = YOLO_DATASET_DIR / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)
(LABELS_DIR / "train").mkdir(parents=True, exist_ok=True)
(LABELS_DIR / "val").mkdir(parents=True, exist_ok=True)
(LABELS_DIR / "test").mkdir(parents=True, exist_ok=True)


# GeoAI
GEOAI_DATASET_DIR = PROCESSED_DIR / "geoai_dataset"
GEOAI_DATASET_DIR.mkdir(parents=True, exist_ok=True)

VECTORS_DIR = GEOAI_DATASET_DIR / "vectors"
VECTORS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_IMG_DIR = GEOAI_DATASET_DIR / "train" / "images"
TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_LABEL_DIR = GEOAI_DATASET_DIR / "train" / "labels"
TRAIN_LABEL_DIR.mkdir(parents=True, exist_ok=True)

TEST_IMG_DIR = GEOAI_DATASET_DIR / "test" / "images"
TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)

TEST_LABEL_DIR = GEOAI_DATASET_DIR / "test" / "labels"
TEST_LABEL_DIR.mkdir(parents=True, exist_ok=True)