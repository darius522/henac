"""Training data must work with current PyTorch and torchaudio releases."""

import numpy as np
import soundfile as sf

from henac.data import RandomAudioDataset


def test_random_excerpts_from_directory_and_csv(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    source = audio_dir / "tone.wav"
    samples = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype("float32")
    sf.write(source, samples, 16000)
    manifest = tmp_path / "files.csv"
    manifest.write_text("path\naudio/tone.wav\n")

    dataset = RandomAudioDataset(
        {"directory": [str(audio_dir)], "manifest": [str(manifest)]},
        sample_rate=32000,
        duration=0.25,
        n_examples=4,
    )
    first = dataset[0]["signal"]
    repeated = dataset[0]["signal"]
    assert first.shape == (1, 1, 8000)
    assert first.sample_rate == 32000
    assert np.array_equal(first.audio_data.numpy(), repeated.audio_data.numpy())
    assert dataset[1]["signal"].shape == first.shape
