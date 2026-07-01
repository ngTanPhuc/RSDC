import geoai.train as gtrain
import wandb
import os


def wandb_patch(output_dir):
    current_epoch = 0
    last_eval_step = -1  # This is used to check if the training is finished and is at the last eval step after the training

    # Initialize the wandb log once, it will automatically check the output folder for the model and log the model
    if wandb.run is not None:
        models_dir = os.path.join(output_dir, "models")
        # W&B will now monitor this path and sync automatically when the file changes
        wandb.save(os.path.join(models_dir, "best_model.pth"), base_path=output_dir, policy="live")
        print("✅ [W&B] Background live-sync enabled for best_model.pth")

    def patched_train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq, scaler=None,
            orig=gtrain.train_one_epoch
    ):
        nonlocal current_epoch
        train_loss = orig(model, optimizer, data_loader, device, epoch, print_freq, scaler)
        current_epoch = epoch + 1

        # Log both scalar and full dict if possible
        wandb.log({
            "epoch": current_epoch,
            "train/loss": float(train_loss)
        }, step=current_epoch)
        print(f"🔗 Epoch {current_epoch}")
        print(f"📊 W&B train loss: {float(train_loss):.4f}")
        return train_loss

    def patched_evaluate(
            model, data_loader, device,
            orig=gtrain.evaluate,
            **kwargs
    ):
        nonlocal current_epoch
        nonlocal last_eval_step
        is_final_eval = (last_eval_step == current_epoch)
        prefix = "final" if is_final_eval else "val"

        # Call original evaluate
        metrics = orig(model, data_loader, device, **kwargs)

        log_dict = {
            "epoch": current_epoch,
            f"{prefix}/loss": metrics.get("loss", 0),
            f"{prefix}/IoU": metrics.get("IoU", 0),
        }

        # Add full COCO metrics
        try:
            coco_metrics = gtrain.evaluate_coco_metrics(
                model, data_loader, device, verbose=False
            )
            log_dict.update({
                f"{prefix}/{k}": v for k, v in coco_metrics.items()
            })
            if is_final_eval: print("[FINAL EVALUATION]")
            print(f"📊 COCO mAP@0.5: {coco_metrics.get('mAP@0.5', 0):.4f} | "
                  f"mAP@0.5:0.95: {coco_metrics.get('mAP@[0.5:0.95]', 0):.4f}")
        except Exception as e:
            print(f"⚠️ COCO metrics failed: {e}")

        wandb.log(log_dict, step=current_epoch)

        last_eval_step = current_epoch
        return metrics

    gtrain.train_one_epoch = patched_train_one_epoch
    gtrain.evaluate = patched_evaluate
    print("🛠️ W&B patches applied")