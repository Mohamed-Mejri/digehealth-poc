from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.data import DigeHealthDataset


def create_dataloaders(
    audio_paths: list[str],
    annotation_paths: list[str],
    train_split_size: float = 0.8,
    batch_size: int = 8,
    num_workers: int = 4,
    random_seed: int = 42,
):
    dataset = DigeHealthDataset(
        audio_paths=audio_paths, annotation_paths=annotation_paths
    )
    generator = torch.Generator().manual_seed(random_seed)

    dataset_size = len(dataset)
    train_size = int(train_split_size * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], generator=generator
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    # Calculate dataset statistics
    train_stats = calculate_dataset_stats(train_dataset)
    test_stats = calculate_dataset_stats(test_dataset)

    return train_loader, test_loader, train_stats, test_stats


def calculate_dataset_stats(dataset) -> dict:
    """Calculate statistics for a dataset"""
    digehealth_dataset = dataset.dataset
    total_frames = sum(
        len(item["labels"]) for item in digehealth_dataset.processed_data
    )
    total_bowel_frames = total_frames - sum(
        (item["labels"] == 0).sum() for item in digehealth_dataset.processed_data
    )

    return {
        "total_files": len(digehealth_dataset.audio_paths),
        "total_chunks": len(digehealth_dataset.processed_data),
        "total_frames": total_frames,
        "total_bowel_frames": total_bowel_frames,
        "bowel_sound_ratio": (
            total_bowel_frames / total_frames if total_frames > 0 else 0
        ),
    }


def get_checkpoint_dir(output_dir: str) -> str:
    """Create a checkpoint directory with today's date, hour, minute, and second"""
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = output_path / "checkpoints" / timestamp
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir)


def cleanup_checkpoints(checkpoint_dir: str):
    """Remove all checkpoint files in the directory"""
    checkpoint_path = Path(checkpoint_dir)
    for checkpoint_file in checkpoint_path.glob("checkpoint_*.pt"):
        try:
            checkpoint_file.unlink()
            print(f"Removed checkpoint: {checkpoint_file.name}")
        except Exception as e:
            print(f"Failed to remove checkpoint {checkpoint_file.name}: {e}")


def load_model_from_checkpoint(
    checkpoint_path: str, model: torch.nn.Module
) -> torch.nn.Module:
    """Load model weights from a checkpoint file"""
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu"), weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
