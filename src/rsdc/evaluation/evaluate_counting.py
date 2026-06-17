from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import torch


def evaluate_counting_yolo(metrics):
    preds_count = []
    gts_count = []

    for img_name, stats in metrics.box.image_metrics.items():
        img_gt_count = stats["tp"] + stats["fn"]

        img_pred_count = stats["tp"] + stats["fp"]

        gts_count.append(img_gt_count)
        preds_count.append(img_pred_count)

    r2 = r2_score(gts_count, preds_count)
    rmse = np.sqrt(mean_squared_error(gts_count, preds_count))

    gts_array = np.array(gts_count)
    preds_array = np.array(preds_count)

    valid_idx = gts_array > 0
    mape = np.mean(np.abs((gts_array[valid_idx] - preds_array[valid_idx]) / gts_array[valid_idx])) * 100

    return r2, rmse, mape

def evaluate_counting_geoai(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    preds_count = []
    gts_count = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            pred_count = len(out["boxes"])
            gt_count = len(tgt["boxes"])
            preds_count.append(pred_count)
            gts_count.append(gt_count)

    r2 = r2_score(gts_count, preds_count)
    rmse = np.sqrt(mean_squared_error(gts_count, preds_count))

    gts_array = np.array(gts_count)
    preds_array = np.array(preds_count)

    valid_idx = gts_array > 0
    mape = np.mean(np.abs((gts_array[valid_idx] - preds_array[valid_idx]) / gts_array[valid_idx])) * 100

    return r2, rmse, mape
