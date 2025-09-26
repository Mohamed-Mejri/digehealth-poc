import logging
import os
from datetime import datetime
from typing import Any

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel

from other.train import cleanup_checkpoints, get_checkpoint_dir
from scripts.train import evaluate_subset_v2, train_epoch
from src.model import DigeHealthModel
from src.utils import create_dataloaders, load_model_from_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__file__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):

    try:
        wandb.init(  # type: ignore
            project=config.wandb.project,
            name=config.wandb.run_name,
            config={
                "batch_size": config.train.batch_size,
                "learning_rate": config.train.learning_rate,
                "num_epochs": config.train.num_epochs,
                "num_workers": config.train.num_workers,
                "model": config.train.hf_model_id,
                "device": config.train.device,
            },
        )
        logger.info("Weights & Biases logging initialized")
        wandb_available = True
    except Exception as e:
        logger.warning(f"Failed to initialize Weights & Biases: {e}")
        logger.warning("Continuing without wandb logging")
        wandb_available = False

    os.makedirs(config.output_dir, exist_ok=True)

    checkpoint_dir = get_checkpoint_dir(config.output_dir)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    logger.info(f"Using device: {config.train.device}")

    logger.info("Creating dataloaders...")
    train_loader, eval_loader, train_stats, eval_stats = create_dataloaders(
        audio_paths=config.data.audio_paths,
        annotation_paths=config.data.annotation_paths,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        random_seed=config.train.random_seed,
        train_split_size=config.train.train_split_size,
    )

    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Eval samples: {len(eval_loader.dataset)}")
    logger.info(f"Using frame rate: {train_loader.dataset.dataset.frame_rate} Hz")

    # Log dataset statistics to wandb
    if wandb_available:
        wandb.log(  # type: ignore
            {
                "dataset/train_samples": len(train_loader.dataset),  # type: ignore
                "dataset/eval_samples": len(eval_loader.dataset),  # type: ignore
                "dataset/train_files": train_stats["total_files"],
                "dataset/train_chunks": train_stats["total_chunks"],
                "dataset/train_frames": train_stats["total_frames"],
                "dataset/train_bowel_frames": train_stats["total_bowel_frames"],
                "dataset/train_bowel_ratio": train_stats["bowel_sound_ratio"],
                "dataset/eval_files": eval_stats["total_files"],
                "dataset/eval_chunks": eval_stats["total_chunks"],
                "dataset/eval_frames": eval_stats["total_frames"],
                "dataset/eval_bowel_frames": eval_stats["total_bowel_frames"],
                "dataset/eval_bowel_ratio": eval_stats["bowel_sound_ratio"],
                "dataset/frame_rate": train_loader.dataset.dataset.frame_rate,  # type: ignore
            }
        )

    if config.checkpoint.load_checkpoint:
        logger.info("Loading model from checkpoint...")
        model = DigeHealthModel(config.train.hf_model_id, config.train.num_labels)
        model = load_model_from_checkpoint(config.checkpoint.checkpoint_path, model)
    else:
        model = DigeHealthModel(config.train.hf_model_id, config.train.num_labels)

    lora_cfg = LoraConfig(
        r=config.lora_config.r,
        lora_alpha=config.lora_config.lora_alpha,
        target_modules=config.lora_config.target_modules,
        lora_dropout=config.lora_config.lora_dropout,
        bias=config.lora_config.bias,
    )

    logger.info("Applying LoRA to the backbone model...")
    model.backbone = get_peft_model(model.backbone, lora_cfg)

    # Ensure classifier is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Only train LoRA params + classifier
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=config.train.learning_rate)
    criterion = nn.CrossEntropyLoss()
    model.to(config.train.device)

    total_params_count = 0
    trainable_params_count = 0
    frozen_params_count = 0
    for name, param in model.named_parameters():
        total_params_count += param.numel()
        if param.requires_grad:
            trainable_params_count += param.numel()
            logger.debug(f"  ✓ Trainable: {name} ({param.numel():,} params)")
        else:
            frozen_params_count += param.numel()
            logger.debug(f"  ❌ Frozen: {name} ({param.numel():,} params)")

    logger.info("Parameter Summary:")
    logger.info(f"  - Total parameters: {total_params_count:,}")
    logger.info(
        f"  - Trainable parameters: {trainable_params_count:,} ({100 * trainable_params_count / total_params_count:.1f}%)"
    )
    logger.info(
        f"  - Frozen parameters: {frozen_params_count:,} ({100 * frozen_params_count / total_params_count:.1f}%)"
    )

    # Log model parameters to wandb
    if wandb_available:
        wandb.log(  # type: ignore
            {
                "model/total_parameters": total_params_count,
                "model/trainable_parameters": trainable_params_count,
                "model/frozen_parameters": frozen_params_count,
                "model/trainable_percentage": 100
                * trainable_params_count
                / total_params_count,
            }
        )

    start_epoch = 0
    global_step = 0
    best_acc = 0.0
    best_f1 = 0.0
    all_step_losses: list[Any] = []

    if wandb_available:
        wandb.log(  # type: ignore
            {
                "training/optimizer": "AdamW",
                "training/learning_rate": config.train.learning_rate,
                "training/criterion": "CrossEntropyLoss",
            }
        )

    # Training loop
    logger.info("Starting training...")

    for epoch in range(start_epoch, config.train.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.train.num_epochs}")

        # Train
        train_loss, train_acc, _ = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.train.device,
            epoch,
            global_step,
            wandb_available,
        )

        eval_loss, eval_acc, eval_precision, eval_f1 = evaluate_subset_v2(
            model, eval_loader, criterion, config.train.device
        )

        logger.info(f"Epoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"  Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}")

        # Log epoch metrics to wandb
        if wandb_available:
            wandb.log(  # type: ignore
                {
                    "epoch": epoch + 1,
                    "train/epoch_loss": train_loss,
                    "train/epoch_accuracy": train_acc,
                    "eval/epoch_loss": eval_loss,
                    "eval/epoch_accuracy": eval_acc,
                    "eval/precision": eval_precision,
                    "eval/f1_score": eval_f1,
                }
            )

        if eval_f1 > best_f1:
            best_f1 = eval_f1

            # Remove any existing checkpoints
            cleanup_checkpoints(checkpoint_dir)

            # Save new best checkpoint
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "best_acc": best_acc,
                "step_losses": all_step_losses,
                "timestamp": datetime.now().isoformat(),
                "model_name": config.train.hf_model_id,
                "learning_rate": config.train.learning_rate,
                "batch_size": config.train.batch_size,
            }
            checkpoint_filename = f"checkpoint_epoch{epoch+1}_step{global_step}.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
            torch.save(checkpoint, checkpoint_path)
            logger.info(
                f"New best checkpoint saved: {checkpoint_filename} (eval_acc: {best_acc:.4f})"
            )

            logger.info(f"New best model saved with accuracy: {best_acc:.4f}")

            # Log new best accuracy to wandb
            if wandb_available:
                wandb.log({"best_accuracy": best_acc})  # type: ignore
        else:
            logger.info(
                f"No new best accuracy. Current best: {best_acc:.4f}, This epoch: {eval_acc:.4f}"
            )

    logger.info(f"Training completed! Best accuracy: {best_acc:.4f}")
    logger.info(f"Model saved to: {config.output_dir}")

    # Log final training summary to wandb
    if wandb_available:
        wandb.log(  # type: ignore
            {
                "training/final_best_accuracy": best_acc,
                "training/total_steps": global_step,
                "training/total_epochs": config.train.num_epochs,
            }
        )

    # Save training history
    training_history = {
        "step_losses": all_step_losses,
        "global_step": global_step,
        "best_acc": best_acc,
    }
    torch.save(training_history, os.path.join(checkpoint_dir, "training_history.pt"))
    logger.info(f"Training history saved to: {checkpoint_dir}/training_history.pt")

    # Finish wandb run
    if wandb_available:
        wandb.finish()  # type: ignore


if __name__ == "__main__":
    main()
