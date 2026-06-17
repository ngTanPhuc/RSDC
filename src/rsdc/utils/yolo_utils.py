import wandb
from pathlib import Path


def save_checkpoint_to_wandb(trainer):
    rank = getattr(trainer, "rank", -1)
    if rank not in (-1, 0):
        return

    epoch_num = int(trainer.epoch) + 1

    print(f"✅ [W&B] on_fit_epoch_end at epoch {epoch_num}, rank={rank}")


    last_path = Path(trainer.last)
    best_path = Path(trainer.best)

    # Use wandb.save() to sync files to the cloud live.
    # This overwrites the previous epoch's weights, preventing storage bloat!
    if wandb.run is not None:
        if last_path.exists():
            wandb.save(str(last_path), base_path=str(last_path.parent))
        if best_path.exists():
            wandb.save(str(best_path), base_path=str(best_path.parent))

    print(f"✅ [W&B] Cloud backup live-synced at epoch {epoch_num}")
