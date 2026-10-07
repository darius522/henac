"""Checkpoint loading and band-wise waveform reconstruction."""

import math
from pathlib import Path

import julius
import numpy as np
import torch
import yaml

from henac.model import DAC


BANDS = ("core", "mb", "hb")


def load_model(
    checkpoint: str | Path, device: str = "cpu", config: str | Path | None = None
) -> DAC:
    """Load a new training checkpoint or a legacy ``dac/weights.pth`` folder."""
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        weights = checkpoint / "dac/weights.pth"
        candidates = (checkpoint / "conf.yaml", checkpoint.parent / "conf.yaml")
    else:
        weights = checkpoint
        candidates = (checkpoint.parent / "conf.yaml",)
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint weights not found: {weights}")
    saved = torch.load(weights, map_location="cpu", weights_only=True)
    if "model_config" in saved:
        model_args = dict(saved["model_config"])
        model_args.setdefault("stage", saved.get("stage", "high"))
    else:
        config_path = (
            Path(config)
            if config is not None
            else next((candidate for candidate in candidates if candidate.is_file()), None)
        )
        if config_path is None:
            raise FileNotFoundError("Legacy checkpoint requires its training conf.yaml")
        loaded = yaml.safe_load(config_path.read_text())
        model_args = {key[4:]: value for key, value in loaded.items() if key.startswith("DAC.")}
        model_args.setdefault("stage", "high")
    model = DAC(**model_args)
    state = saved.get("model_state", saved.get("state_dict"))
    if state is None:
        raise ValueError("Checkpoint has no model_state or state_dict")
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def codebook_limits(model: DAC) -> tuple[int, int, int]:
    counts = [model.n_codebooks] + [skip.n_codebooks for skip in model.skip_aes]
    return tuple((counts + [0, 0])[:3])


def validate_codebooks(model: DAC, counts: tuple[int, int, int]) -> None:
    if len(counts) != 3:
        raise ValueError("Specify core, mid, and high codebook counts")
    limits = codebook_limits(model)
    if not 1 <= counts[0] <= limits[0]:
        raise ValueError(f"Core codebooks must be in [1, {limits[0]}]")
    if not 0 <= counts[1] <= limits[1]:
        raise ValueError(f"Mid codebooks must be in [0, {limits[1]}]")
    if not 0 <= counts[2] <= limits[2] or (counts[2] and not counts[1]):
        raise ValueError("High codebooks require an active mid path")


def aligned_length(model: DAC, length: int, counts: tuple[int, int, int]) -> int:
    """Round up to a length supported by every requested decoder path."""
    if length <= 0:
        raise ValueError("Input audio must contain at least one sample")
    alignment = model.hop_length
    active_skips = 2 if counts[2] else 1 if counts[1] else 0
    for index, skip in enumerate(model.skip_aes[:active_skips]):
        feature_hop = math.prod(model.encoder_rates[: -(index + 1)])
        alignment = math.lcm(alignment, feature_hop * math.prod(skip.encoder_rates))
    return math.ceil(length / alignment) * alignment


def combine_bands(
    paths: dict[str, torch.Tensor], sample_rate: int, output_length: int
) -> np.ndarray:
    """Filter decoder paths at the reference inference cutoffs and sum them."""
    active = tuple(name for name in BANDS if name in paths)
    if active == ("core",):
        return paths["core"][..., :output_length].cpu().numpy().reshape(-1)
    cutoffs = [3000] if active == ("core", "mb") else [3000, 6200]
    splitter = julius.SplitBands(sample_rate, cutoffs=cutoffs).to(paths["core"].device)
    mixed = np.zeros((1, 1, output_length))
    for band, name in enumerate(active):
        filtered = splitter(paths[name])[band][..., :output_length]
        mixed += filtered.reshape(1, 1, -1).cpu().numpy()
    return mixed.reshape(-1)


@torch.no_grad()
def reconstruct(
    model: DAC, audio: torch.Tensor, counts: tuple[int, int, int]
) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    """Reconstruct mono ``[1, 1, T]`` audio and return path codes."""
    validate_codebooks(model, counts)
    if audio.ndim != 3 or audio.shape[:2] != (1, 1):
        raise ValueError("Expected mono audio with shape [1, 1, samples]")
    length = audio.shape[-1]
    padded_length = aligned_length(model, length, counts)
    if padded_length != length:
        audio = torch.nn.functional.pad(audio, (0, padded_length - length))
    if counts[2] > 0:
        result = model.infer_bands(audio, n_quantizers=list(counts))
    else:
        result = model.infer_active_bands(audio, counts)
    waveform = combine_bands(result["audio"], model.sample_rate, length)
    return waveform, result["codes"]


def empirical_bitrate(
    codes: dict[str, torch.Tensor], duration_seconds: float, codebook_size: int
) -> dict[str, float]:
    """Estimate bits per second from observed code histograms, without coding."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    rates = {}
    for name, path_codes in codes.items():
        values = path_codes.detach().cpu().numpy()
        bits = 0.0
        for book in values[0]:
            histogram = np.bincount(book.astype(np.int64), minlength=codebook_size)
            probabilities = histogram[histogram > 0] / histogram.sum()
            bits += float(-(probabilities * np.log2(probabilities)).sum()) * len(book)
        rates[name] = bits / duration_seconds
    rates["total"] = sum(rates.values())
    return rates
