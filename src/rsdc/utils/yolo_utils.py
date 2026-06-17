import wandb
from pathlib import Path
from configs.yolo_hyperparams import hyperparams

def save_checkpoint_to_wandb(trainer):
    rank = getattr(trainer, "rank", -1)
    if rank not in (-1, 0):
        return

    epoch_num = int(trainer.epoch) + 1

    print(f"✅ [W&B] on_fit_epoch_end at epoch {epoch_num}, rank={rank}")

    artifact = wandb.Artifact(
        name=hyperparams["name"],
        type="model",
        metadata={"epoch": epoch_num}
    )

    last_path = Path(trainer.last)
    best_path = Path(trainer.best)

    print(f"✅ [W&B] last exists: {last_path.exists()} -> {last_path}")
    print(f"✅ [W&B] best exists: {best_path.exists()} -> {best_path}")

    added = 0

    if last_path.exists():
        artifact.add_file(str(last_path), name=f"last_epoch_{epoch_num}.pt")
        print("✅ added last.pt")
        added += 1

    if best_path.exists():
        artifact.add_file(str(best_path), name=f"best_epoch_{epoch_num}.pt")
        print("✅ added best.pt")
        added += 1

    print(f"✅ files added = {added}")

    if added > 0 and wandb.run is not None:
        wandb.run.log_artifact(artifact)
        print("✅ artifact logged")
    else:
        print("❌ no files added, skip log_artifact")
