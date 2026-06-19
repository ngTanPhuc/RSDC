from pathlib import Path

from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import torch


def evaluate_counting_yolo(metrics, print_counts=True):
    rows = []
    preds_count = []
    gts_count = []

    for img_name, stats in metrics.box.image_metrics.items():
        img_gt_count = int(stats["tp"] + stats["fn"])
        img_pred_count = int(stats["tp"] + stats["fp"])
        error = img_pred_count - img_gt_count

        rows.append({
            "image": img_name,
            "gt_count": img_gt_count,
            "pred_count": img_pred_count,
            "error": error,
        })

        gts_count.append(img_gt_count)
        preds_count.append(img_pred_count)

    r2, rmse, mape = calculate_metrics(gts_count, preds_count)

    if print_counts:
        print_counting_table(rows)

    return r2, rmse, mape, rows


def evaluate_counting_geoai(
    model,
    data_loader,
    score_threshold=0.2,
    print_counts=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    preds_count = []
    gts_count = []

    image_paths = getattr(data_loader.dataset, "image_paths", None)
    batch_start = 0

    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for local_idx, (out, tgt) in enumerate(zip(outputs, targets)):
                global_idx = batch_start + local_idx
                image_name = Path(image_paths[global_idx]).name if image_paths else str(global_idx)

                pred_count = int((out["scores"] >= score_threshold).sum().item())
                gt_count = int(len(tgt["boxes"]))
                error = pred_count - gt_count

                rows.append({
                    "image": image_name,
                    "gt_count": gt_count,
                    "pred_count": pred_count,
                    "error": error,
                })

                preds_count.append(pred_count)
                gts_count.append(gt_count)

            batch_start += len(images)

    r2, rmse, mape = calculate_metrics(gts_count, preds_count)

    if print_counts:
        print_counting_table(rows)

    return r2, rmse, mape, rows


def calculate_metrics(gts_count, preds_count):
    if len(gts_count) == 0:
        return np.nan, np.nan, np.nan

    gts_array = np.array(gts_count, dtype=float)
    preds_array = np.array(preds_count, dtype=float)

    r2 = r2_score(gts_array, preds_array)
    rmse = np.sqrt(mean_squared_error(gts_array, preds_array))

    valid_idx = gts_array > 0
    if np.any(valid_idx):
        mape = np.mean(
            np.abs((gts_array[valid_idx] - preds_array[valid_idx]) / gts_array[valid_idx])
        ) * 100
    else:
        mape = 0.0

    return r2, rmse, mape


def print_counting_table(rows):
    if not rows:
        print("[Counting] No rows to print.")
        return

    columns = ["image", "gt_count", "pred_count", "error"]
    widths = {
        "image": max(len("image"), *(len(str(row["image"])) for row in rows)),
        "gt_count": max(len("gt_count"), *(len(str(row["gt_count"])) for row in rows)),
        "pred_count": max(len("pred_count"), *(len(str(row["pred_count"])) for row in rows)),
        "error": max(len("error"), *(len(str(row["error"])) for row in rows)),
    }

    header = " | ".join(col.ljust(widths[col]) for col in columns)
    separator = "-+-".join("-" * widths[col] for col in columns)

    print("\n[Counting] Per-image predicted vs ground-truth counts:")
    print(header)
    print(separator)

    for row in rows:
        values = [
            str(row["image"]).ljust(widths["image"]),
            str(row["gt_count"]).rjust(widths["gt_count"]),
            str(row["pred_count"]).rjust(widths["pred_count"]),
            str(row["error"]).rjust(widths["error"]),
        ]
        print(" | ".join(values))

    print()
