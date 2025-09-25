import torch
from torch.utils.data import DataLoader, random_split

from data import DigeHealthDataset


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
    test_stats = calculate_dataset_stats(test_loader)

    return train_loader, test_loader, train_stats, test_stats


def calculate_dataset_stats(dataset) -> dict:
    """Calculate statistics for a dataset"""
    total_frames = sum(len(item["labels"]) for item in dataset.processed_data)
    total_bowel_frames = total_frames - sum(
        (item["labels"] == 0).sum() for item in dataset.processed_data
    )

    return {
        "total_files": len(dataset.audio_paths),
        "total_chunks": len(dataset.processed_data),
        "total_frames": total_frames,
        "total_bowel_frames": total_bowel_frames,
        "bowel_sound_ratio": (
            total_bowel_frames / total_frames if total_frames > 0 else 0
        ),
    }
