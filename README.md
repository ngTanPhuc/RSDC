# RSDC — Rice Seedlings Detection & Counting

Detect and count rice seedlings from 4-band aerial/satellite imagery (RGB + Near-Infrared) using two independent deep learning pipelines: **Mask R-CNN / Faster R-CNN** (via `geoai-py`) and **YOLO26m** (via `ultralytics`).

---

## Project Status

| Component | Status |
|-----------|--------|
| Data preparation scripts | Complete |
| GeoAI training pipeline (Mask R-CNN / Faster R-CNN) | Complete |
| YOLO26m training pipeline | Complete |
| Counting evaluation (R², RMSE, MAPE) | Complete |
| W&B experiment tracking | Integrated |
| Test suite | Missing |
| `pyproject.toml`, `environment.yml`, `README.md` | Stubs — no content |
| `reports/` | Empty (not yet generated) |

**Known issues:**
- `configs/geoai_hyperparams.py` omits `num_epochs`; it is injected at runtime by `run_geoai_train_val.py:63`.
- `configs/yolo_hyperparams.py` sets `num_channels: 4`, but YOLO uses 3-channel RGB PNGs — this field is unused.
- `scripts/prepare_yolo_dataset.py:76` (`clip_box_2_tile`) rejects bounding boxes touching tile edges, discarding partially visible seedlings.
- No automated tests exist.

---

## Requirements

```txt
ultralytics
rasterio
tqdm
pillow
numpy
wandb
scikit-learn
geopandas
geoai-py
```

Install with:

```bash
pip install -r requirements.txt
```

A Weights & Biases account is required for experiment tracking. Set your API key:

```bash
export WANDB_API_KEY=your_key_here
```

---

## Dataset

The dataset consists of **8 aerial/satellite images** (1527×1527 px, 4-band GeoTIFF: R, G, B, NIR) with **PASCAL VOC XML annotations** for a single class: `RiceSeedling`.

### Download

```bash
bash scripts/download_dataset
```

This runs:

```bash
kaggle datasets download -d ngtanphuc020505/rice-seedlings-dataset -p data/raw --unzip
```

Expected structure after download:

```
data/raw/
├── rasters/        # 1.tif .. 8.tif
└── annotations/    # 1.xml .. 8.xml, classes.txt
```

---

## End-to-End Training — GeoAI Pipeline (Mask R-CNN / Faster R-CNN)

### Step 1: Prepare the dataset

```bash
python scripts/prepare_geoai_dataset.py
```

This produces `data/processed/geoai_dataset/`:
- **Tiles:** 512×512 px, 4-band GeoTIFF, stride 256
- **Split:** images 1–6 → train, images 7–8 → test
- **Labels:** rasterized masks + GeoJSON vectors

### Step 2: Train

```bash
python scripts/run_geoai_train_val.py <output_dir> <base_model> [-e EPOCHS] [--score-threshold THRESH]
```

**Arguments:**

| Argument             | Description                                      | Default     |
|----------------------|--------------------------------------------------|-------------|
| `output_dir`         | Subdirectory name under `experiments/`           | (required)  |
| `base_model`         | `maskrcnn` or `fasterrcnn`                       | (required)  |
| `-e` / `--epochs`    | Number of training epochs                        | `150`       |
| `--score-threshold`  | Confidence threshold for counting evaluation     | `0.2`       |

Example:

```bash
python scripts/run_geoai_train_val.py geoai_run_1 maskrcnn -e 200 --score-threshold 0.3
```

Results are saved to `experiments/geoai_run_1/`.

### Step 3: What is evaluated

After training, the script automatically:
1. Loads `best_model.pth`
2. Runs test inference — reports **IoU** and **COCO mAP** metrics
3. Computes **counting metrics** (R², RMSE, MAPE) with a per-image comparison table

---

## End-to-End Training — YOLO Pipeline (YOLO26m)

### Step 1: Prepare the dataset

```bash
python scripts/prepare_yolo_dataset.py
```

This produces `data/processed/yolo_dataset/`:
- **Tiles:** 640×640 px, 3-band RGB PNG, stride 320
- **Split:** images 1–5 → train, image 6 → val, images 7–8 → test
- **Labels:** YOLO-format `.txt` files (`class cx cy w h`)

### Step 2: Train

```bash
python scripts/run_yolo26m_train_val.py <output_dir> [-e EPOCHS] [--conf THRESH]
```

**Arguments:**

| Argument          | Description                                      | Default     |
|-------------------|--------------------------------------------------|-------------|
| `output_dir`      | Subdirectory name under `experiments/`           | (required)  |
| `-e` / `--epochs` | Number of training epochs                        | `150`       |
| `--conf`          | Confidence threshold for validation and counting | `0.25`      |

Example:

```bash
python scripts/run_yolo26m_train_val.py yolo_run_1 -e 200 --conf 0.3
```

Results are saved to `experiments/yolo_run_1/`.

### Step 3: What is evaluated

After training, the script automatically:
1. Runs validation on the **test set** (`configs/yolo_test_file.yaml`)
2. Reports **mAP50-95**, **mAP50**, **mAP75**
3. Extracts per-image TP/FP/FN from YOLO metrics
4. Computes **counting metrics** (R², RMSE, MAPE) with a per-image comparison table

---

## Training Augmentations

### GeoAI
Handled internally by `geoai-py` — standard torchvision transforms for detection.

### YOLO26m (`src/rsdc/training/train_yolo26m.py`)
| Augmentation | Value |
|-------------|-------|
| Mosaic | `0.8` |
| MixUp | `0.10` |
| Scale | `0.3` |
| Flip LR | `0.35` |
| Flip UD | `0.4` |
| HSV Saturation | `0.4` |
| HSV Value | `0.2` |
| Shear | `5°` |
| Close mosaic epoch | `10` |
| Early stopping patience | `20` |

---

## Counting Metrics

Both pipelines use shared counting evaluation (`src/rsdc/evaluation/evaluate_counting.py`):

- **Predicted count** = TP + FP
- **Ground truth count** = TP + FN
- **R²** — coefficient of determination (`sklearn.metrics.r2_score`)
- **RMSE** — root mean squared error
- **MAPE** — mean absolute percentage error (excludes images with zero ground-truth count)

A formatted per-image table is printed showing image name, ground truth count, predicted count, and error.

---

## Experiment Tracking (Weights & Biases)

Both pipelines log to W&B:

| Pipeline | Logged metrics | Checkpoint sync |
|----------|---------------|-----------------|
| **GeoAI** | Train loss, val loss, IoU, COCO mAP per epoch | `best_model.pth` after each epoch |
| **YOLO** | Ultralytics default metrics per epoch | `last.pt` + `best.pt` after each epoch |

---

## Project Structure

```
RSDC/
├── configs/              # Paths, hyperparameters, YAML dataset configs
├── scripts/              # Data preparation + training entry points
├── src/rsdc/
│   ├── training/         # Model training wrappers
│   ├── evaluation/       # Shared counting evaluation
│   └── utils/            # W&B logging patches and callbacks
├── data/                 # Raw + processed datasets (gitignored)
├── models/               # Trained model checkpoints (gitignored)
├── experiments/          # Training run outputs (gitignored)
├── notebooks/            # Data exploration and error analysis
├── reports/              # Generated reports and figures
└── PROJECT_ANALYSIS.md   # Detailed architecture and code analysis
```

---

## Citation

If you use this project, cite the dataset:

```
@misc{rice-seedlings-dataset,
  author = {Nguyen Tan Phuc},
  title  = {Rice Seedlings Dataset},
  year   = {2024},
  publisher = {Kaggle},
  url    = {https://www.kaggle.com/datasets/ngtanphuc020505/rice-seedlings-dataset}
}
```
