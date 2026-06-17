from pathlib import Path

# 1. Dynamically find the project root
# __file__ is the absolute path to this specific paths.py file
# .resolve() resolves any symlinks
# .parents[1] goes up two levels (from configs/paths.py -> configs/ -> RSDC/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 2. Define standard project directories
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATION_DIR = RAW_DIR / "annotations"
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
RASTER_DIR = RAW_DIR / "rasters"
RASTER_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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
