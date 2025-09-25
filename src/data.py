import logging
import math
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

FRAME_RATE = 49.5
STRIDE_MS = 20
RECEPTIVE_FIELD_MS = 25
ANNOTATION_EXTENTION = ".txt"
AUDIO_EXTENTION = ".wav"
COLUMNS = ["start", "end", "label"]
LABEL_MAP = {
    "background": 0,
    "b": 1,
    "mb": 2,
    "h": 3,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__file__)


class DigeHealthDataset(Dataset):
    def __init__(
        self,
        audio_paths: list[str],
        annotation_paths: list[str],
        max_length=2.0,
        sample_rate=16000,
    ):
        self.max_length = max_length
        self.sample_rate = sample_rate

        self.frame_rate = FRAME_RATE

        self.stride_ms = STRIDE_MS
        self.receptive_field_ms = RECEPTIVE_FIELD_MS
        self.annotation_paths = annotation_paths
        self.audio_paths = audio_paths
        self.audio_chunk_duration = int(self.max_length * self.sample_rate)
        self.chunk_duration_frames = int(self.max_length * self.frame_rate)

        self.processed_data = self._load_and_process_data()

    def _clean_label(self, label):
        label = label.strip()
        if label in ["sb", "sbs"]:
            label = "b"
        return label

    def _map_label(self, label):
        return LABEL_MAP[label]

    def _process_annotation(self, data):
        data = data[data["label"].isin(LABEL_MAP.keys())]
        data.label = data.label.apply(lambda x: self._map_label(x))
        return data

    def _load_and_process_data(self):
        # Track some statistics for debugging

        processed_data = []
        total_annotations = 0
        total_bowel_sound_frames = 0
        sample_labels_shown = 0

        for wav_path, annotation_path in tqdm(
            zip(self.audio_paths, self.annotation_paths)
        ):
            # Load audio
            try:
                audio, sr = librosa.load(wav_path, sr=self.sample_rate)
            except Exception as e:
                logger.error(f"Error loading {wav_path}: {e}")
                continue

            # Load annotations
            try:
                annotations = pd.read_csv(
                    annotation_path, names=["start", "end", "label"], sep="\t"
                )
                annotations.label = annotations.label.apply(
                    lambda x: self._clean_label(x)
                )
                annotations = self._process_annotation(annotations)
            except Exception as e:
                logger.error(f"Error loading {annotation_path}: {e}")
                continue

            # Calculate number of frames for this audio
            audio_duration = len(audio) / self.sample_rate
            num_frames = int(audio_duration * self.frame_rate)

            logger.info(f"audio duration: {audio_duration}, frames: {num_frames}")

            # Initialize labels as zeros
            labels = np.zeros(num_frames, dtype=np.int64)
            file_annotations = 0

            for idx, row in annotations.iterrows():
                if pd.isna(row["start"]) or pd.isna(row["end"]):  # type: ignore
                    continue

                start_frame = self._time_to_frame_index(row["start"])
                end_frame = self._time_to_frame_index(row["end"])
                label = row["label"]

                labels[start_frame : end_frame + 1] = label
                file_annotations += 1
                total_bowel_sound_frames += end_frame - start_frame + 1

            total_annotations += file_annotations

            chunk_id = -1
            for i in tqdm(range(0, len(labels), self.chunk_duration_frames)):
                chunk_id += 1
                chunk_labels = labels[i : i + self.chunk_duration_frames]
                chunk_audio = audio[
                    chunk_id
                    * self.audio_chunk_duration : (chunk_id + 1)
                    * self.audio_chunk_duration
                ]

                if i + self.chunk_duration_frames > len(labels):
                    chunk_labels = np.pad(
                        chunk_labels,
                        (0, self.chunk_duration_frames - len(chunk_labels)),
                        "constant",
                    )
                    chunk_audio = np.pad(
                        chunk_audio,
                        (0, self.audio_chunk_duration - len(chunk_audio)),
                        "constant",
                    )
                processed_data.append(
                    {
                        "audio": chunk_audio,
                        "labels": chunk_labels,
                        "filename": Path(wav_path).stem,
                        "chunk_idx": chunk_id,
                    }
                )
        return processed_data

    def _time_to_frame_index(self, time_seconds):
        """Convert time in seconds to frame index at 49Hz"""
        return math.ceil(time_seconds * self.frame_rate)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]
        return {
            "audio": torch.FloatTensor(item["audio"]),
            "labels": torch.LongTensor(item["labels"]),
            "filename": item["filename"],
            "chunk_idx": item["chunk_idx"],
        }
