import geoai.train as gtrain
import wandb
import os


def wandb_patch(output_dir):
    current_epoch = 0

    def patched_train_one_epoch(
            model, optimizer, data_loader, device, epoch, print_freq, scaler=None,
            orig=gtrain.train_one_epoch
    ):
        nonlocal current_epoch
        train_loss = orig(model, optimizer, data_loader, device, epoch, print_freq, scaler)
        current_epoch = epoch + 1

        # Log both scalar and full dict if possible
        wandb.log({
            "epoch": epoch,
            "train/loss": float(train_loss)
        }, step=epoch)
        print(f"🔗 Epoch {current_epoch}")
        print(f"📊 W&B train loss: {float(train_loss):.4f}")
        return train_loss

    def patched_evaluate(
            model, data_loader, device,
            orig=gtrain.evaluate,
            **kwargs
    ):
        nonlocal current_epoch

        # Call original evaluate
        metrics = orig(model, data_loader, device, **kwargs)

        log_dict = {
            "epoch": current_epoch,
            "val/loss": metrics.get("loss", 0),
            "val/IoU": metrics.get("IoU", 0),
        }

        # Add full COCO metrics
        try:
            coco_metrics = gtrain.evaluate_coco_metrics(
                model, data_loader, device, verbose=False
            )
            log_dict.update({
                f"val/{k}": v for k, v in coco_metrics.items()
            })
            print(f"📊 COCO mAP@0.5: {coco_metrics.get('mAP@0.5', 0):.4f} | "
                  f"mAP@0.5:0.95: {coco_metrics.get('mAP@[0.5:0.95]', 0):.4f}")
        except Exception as e:
            print(f"⚠️ COCO metrics failed: {e}")

        wandb.log(log_dict, step=current_epoch)

        if wandb.run is not None:
            models_dir = os.path.join(output_dir, "models")
            best_ckpt_path = os.path.join(models_dir, "best_model.pth")

            if os.path.exists(best_ckpt_path):
                # Sync to the cloud, overwriting the previous epoch's file
                wandb.save(best_ckpt_path, base_path=output_dir)
                print(f"✅ [W&B] Cloud backup live-synced at epoch {current_epoch}")

        return metrics

    gtrain.train_one_epoch = patched_train_one_epoch
    gtrain.evaluate = patched_evaluate
    print("🛠️ W&B patches applied")