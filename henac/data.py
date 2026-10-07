"""Random audio excerpts for reproducible HENAC training."""

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from audiotools import AudioSignal
from audiotools.core import util
from torch.utils.data import Dataset


EXTENSIONS = {".wav", ".flac"}


def _source_files(source: str | Path) -> list[Path]:
    source = Path(source)
    if source.is_dir():
        paths = sorted(path for path in source.rglob("*") if path.suffix.lower() in EXTENSIONS)
    elif source.suffix.lower() == ".csv":
        with source.open(newline="") as file:
            rows = list(csv.reader(file))
        if rows and rows[0] and rows[0][0].lower() == "path":
            rows = rows[1:]
        paths = [Path(row[0]) for row in rows if row and row[0]]
        paths = [path if path.is_absolute() else source.parent / path for path in paths]
    elif source.is_file() and source.suffix.lower() in EXTENSIONS:
        paths = [source]
    else:
        raise FileNotFoundError(f"Audio source not found: {source}")
    if not paths:
        raise ValueError(f"No WAV or FLAC files in {source}")
    missing = next((path for path in paths if not path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(f"Audio path from {source} does not exist: {missing}")
    return paths


class RandomAudioDataset(Dataset):
    """Draw fixed-length mono excerpts, cycling evenly through source groups."""

    def __init__(
        self,
        folders: dict[str, list[str]],
        sample_rate: int,
        duration: float,
        n_examples: int,
        transform=None,
        seed: int = 0,
    ):
        if duration <= 0 or n_examples <= 0:
            raise ValueError("duration and n_examples must be positive")
        self.groups = [
            [audio for source in sources for audio in _source_files(source)]
            for sources in folders.values()
        ]
        if not self.groups:
            raise ValueError("At least one audio source group is required")
        if any(not group for group in self.groups):
            raise ValueError("Each audio source group must contain at least one file")
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_examples = n_examples
        self.transform = transform
        self.seed = seed

    def __len__(self):
        return self.n_examples

    def __getitem__(self, index):
        state = util.random_state(self.seed + index)
        group = self.groups[index % len(self.groups)]
        path = group[state.randint(len(group))]
        info = sf.info(path)
        source_frames = round(self.duration * info.samplerate)
        offset = (
            state.randint(info.frames - source_frames + 1) if info.frames > source_frames else 0
        )
        with sf.SoundFile(path) as file:
            file.seek(offset)
            samples = file.read(source_frames, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(np.asarray(samples.T).copy()).mean(0, keepdim=True)
        if info.samplerate != self.sample_rate:
            waveform = audio_functional.resample(waveform, info.samplerate, self.sample_rate)
        target_frames = round(self.duration * self.sample_rate)
        waveform = torch.nn.functional.pad(
            waveform[..., :target_frames], (0, max(0, target_frames - waveform.shape[-1]))
        )
        signal = AudioSignal(waveform.unsqueeze(0), self.sample_rate)
        result = {"signal": signal, "idx": index}
        if self.transform is not None:
            result["transform_args"] = self.transform.instantiate(state=state, signal=signal)
        return result

    @staticmethod
    def collate(items):
        return util.collate(items)
