import logging
import os
from datetime import datetime
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from sklearn.metrics import f1_score, precision_score
from tqdm import tqdm

import wandb
from src.model import DigeHealthModel
from src.utils import cleanup_checkpoints, create_dataloaders, get_checkpoint_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__file__)


def evaluate_subset_v2(model, eval_loader, criterion, device, max_batches=10):
    """Evaluate on a subset of the evaluation set for faster evaluation during training"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            if num_batches >= max_batches:
                break

            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(audio)
            logits = outputs.logits

            batch_size, seq_len, num_labels = logits.shape

            # Pad/truncate labels to match logits
            if labels.size(1) != seq_len:
                if labels.size(1) > seq_len:
                    labels = labels[:, :seq_len]
                else:
                    padding = torch.zeros(
                        batch_size,
                        seq_len - labels.size(1),
                        dtype=labels.dtype,
                        device=labels.device,
                    )
                    labels = torch.cat([labels, padding], dim=1)

            # Flatten for loss & metric calculation
            logits_flat = logits.view(-1, num_labels)
            labels_flat = labels.view(-1)

            # Loss
            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item()

            # Predictions
            predictions = torch.argmax(logits_flat, dim=1)

            # Store predictions and labels for metrics
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels_flat.cpu().numpy())

            num_batches += 1

    # Compute accuracy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    # Optionally ignore padded labels if you used -100 for padding
    mask = all_labels != 0  # adjust if you used a different ignore index
    all_preds = all_preds[mask]
    all_labels = all_labels[mask]

    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average="macro")
    precision = precision_score(all_labels, all_preds, average="macro")

    return total_loss / num_batches, accuracy, precision, f1


def train_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    epoch=None,
    global_step=0,
    wandb_available=False,
):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    step_losses = []

    progress_bar = tqdm(train_loader, desc="Training Pipeline")

    for batch in progress_bar:
        audio = batch["audio"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(audio)
        logits = outputs.logits  # shape: (batch_size, sequence_length, num_labels)

        batch_size, seq_len, num_labels = logits.shape

        # for loss calculation
        logits_flat = logits.view(-1, num_labels)
        labels_flat = labels.view(-1)

        # calculate loss
        loss = criterion(logits_flat, labels_flat)

        loss.backward()

        optimizer.step()

        predictions = torch.argmax(logits_flat)
        correct_predictions += (predictions == labels_flat).sum().item()
        total_predictions += labels_flat.size(0)

        total_loss += loss.item()
        step_losses.append(loss.item())  # Store step loss

        # Log step metrics to wandb
        current_step = global_step + progress_bar.n
        if wandb_available:
            wandb.log(
                {
                    "train/step_loss": loss.item(),
                    "train/step_accuracy": correct_predictions / total_predictions,
                    "train/step": current_step,
                }
            )

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * correct_predictions / total_predictions:.2f}%",
            }
        )

    return (
        total_loss / len(train_loader),
        correct_predictions / total_predictions,
        step_losses,
    )


def evaluate_subset(model, eval_loader, criterion, device, max_batches=10):
    """Evaluate on a subset of the evaluation set for faster evaluation during training"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    num_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            if num_batches >= max_batches:
                break

            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(audio)
            logits = outputs.logits

            # Get the actual sequence length from model output
            batch_size, seq_len, num_labels = logits.shape

            # Ensure labels match the model's output sequence length
            if labels.size(1) != seq_len:
                # Pad or truncate labels to match model output
                if labels.size(1) > seq_len:
                    # Truncate labels
                    labels = labels[:, :seq_len]
                else:
                    # Pad labels with zeros
                    padding = torch.zeros(
                        batch_size,
                        seq_len - labels.size(1),
                        dtype=labels.dtype,
                        device=labels.device,
                    )
                    labels = torch.cat([labels, padding], dim=1)

            # Reshape for loss calculation
            logits_flat = logits.view(-1, num_labels)
            labels_flat = labels.view(-1)

            # Calculate loss
            loss = criterion(logits_flat, labels_flat)

            # Calculate accuracy
            predictions = torch.argmax(logits_flat, dim=1)
            correct_predictions += (predictions == labels_flat).sum().item()
            total_predictions += labels_flat.size(0)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches, correct_predictions / total_predictions


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

    # Create checkpoint directory for today
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

    model = DigeHealthModel(config.train.hf_model_id, config.train.num_labels)
    model.freeze_backbone()

    logger.debug("Parameter training status:")
    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            logger.debug(f"  ✓ Trainable: {name} ({param.numel():,} params)")
        else:
            frozen_params += param.numel()
            logger.debug(f"  ❌ Frozen: {name} ({param.numel():,} params)")

    logger.info("Parameter Summary:")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(
        f"  - Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)"
    )
    logger.info(
        f"  - Frozen parameters: {frozen_params:,} ({100 * frozen_params / total_params:.1f}%)"
    )

    # Log model parameters to wandb
    if wandb_available:
        wandb.log(  # type: ignore
            {
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/frozen_parameters": frozen_params,
                "model/trainable_percentage": 100 * trainable_params / total_params,
            }
        )

    model = model.to(config.train.device)

    optimizer = optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    global_step = 0
    best_acc = 0.0
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
        train_loss, train_acc, step_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.train.device,
            epoch,
            global_step,
            wandb_available,
        )

        # #  Log step losses for this epoch
        # for step_loss in step_losses:
        #     global_step += 1
        #     all_step_losses.append(step_loss)
        #     logger.debug(f"Step {global_step}: loss = {step_loss:.4f}")

        #     # Evaluate periodically
        #     if global_step % config..eval_steps == 0:
        #         eval_loss, eval_acc = evaluate_subset(
        #             model, eval_loader, criterion, device
        #         )
        #         logger.debug(
        #             f"Step {global_step} - Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}"
        #         )
        #         # Log evaluation metrics to wandb
        #         if wandb_available:
        #             wandb.log(
        #                 {
        #                     "eval/step_loss": eval_loss,
        #                     "eval/step_accuracy": eval_acc,
        #                     "eval/step": global_step,
        #                 }
        #             )

        # Evaluation at end of epoch (using subset of test data for efficiency)
        # eval_loss, eval_acc = evaluate_subset(
        #     model, eval_loader, criterion, config.train.device
        # )

        eval_loss, eval_acc, precision, f1 = evaluate_subset_v2(
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
                    "eval/precision": precision,
                    "eval/f1_score": f1,
                }
            )

        # Save checkpoint only if this is the best eval_acc so far in this run
        if eval_acc > best_acc:
            best_acc = eval_acc

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

            # Save best model
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model"))
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
