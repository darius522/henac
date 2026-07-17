import csv
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import sys

import argbind
import torch, torchaudio
from audiotools import AudioSignal
from audiotools import metrics
from audiotools.core import util
from audiotools.ml.decorators import Tracker
from train import losses
try:
    from GOMPSNR.GOMPSNR import GOMPSNR
except ModuleNotFoundError:
    # Allow running as `python scripts/evaluate.py` where project root
    # is not automatically added to sys.path.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from GOMPSNR.GOMPSNR import GOMPSNR

import julius


@dataclass
class State:
    stft_loss: losses.BandpassedMultiScaleSTFTLoss
    mel_loss: losses.BandpassedMelSpectrogramLoss
    waveform_loss: losses.L1Loss
    sisdr_loss: losses.SISDRLoss
    gompsnr: GOMPSNR
    selected_metrics: tuple


METRIC_ALIASES = {
    "gomsnp": "gompsnr",
}


def parse_metrics(metrics: str) -> tuple:
    valid_metrics = {"mel", "stft", "waveform", "sisdr", "visqol", "gompsnr"}
    selected = []
    for metric in [m.strip().lower() for m in metrics.split(",") if m.strip()]:
        normalized = METRIC_ALIASES.get(metric, metric)
        if normalized not in valid_metrics:
            raise ValueError(
                f"Unsupported metric '{metric}'. Valid metrics: {sorted(valid_metrics)}"
            )
        selected.append(normalized)

    if not selected:
        raise ValueError("No metrics selected. Provide at least one metric name.")

    # Keep user-defined order while removing duplicates.
    return tuple(dict.fromkeys(selected))

def percent_silent(filepath, threshold=1e-3, frame_size=1024, hop=512):
    wav, sr = torchaudio.load(filepath)          # [C, T]
    wav = wav.mean(0)                            # mono
    
    # Frame RMS
    frames = wav.unfold(0, frame_size, hop)      # [num_frames, frame_size]
    rms = torch.sqrt((frames ** 2).mean(dim=1))  # [num_frames]

    # Percent of frames considered silent
    silent = (rms < threshold).float().mean().item()
    return silent

def get_metrics(signal_path, recons_path, state, idx):
    output = {}
    bands = [3000, 6000]
    selected_metrics = set(state.selected_metrics)
    signal = AudioSignal(signal_path)
    recons = AudioSignal(recons_path)
    sp = julius.SplitBands(32_000, cutoffs=[b for b in bands]) # slight overlap since rolloff is not brickwall
    xb = torch.concat([sp(signal.audio_data.clone()), torch.sum(sp(signal.audio_data.clone()), 0, keepdim=True)], 0)
    yb = torch.concat([sp(recons.audio_data.clone()), torch.sum(sp(recons.audio_data.clone()), 0, keepdim=True)], 0)
    for i, (x, y, k) in enumerate(zip(xb, yb, bands + [16000, 'full'])):
        print(f'proc: {idx}')
        x = AudioSignal(x, signal.sample_rate)
        y = AudioSignal(y, signal.sample_rate)

        # print()
        fmin = 0 if k in [3000, 'full'] else bands[i - 1]
        fmax = int(k) if k != 'full' else None
        if "mel" in selected_metrics:
            output[f"mel-{k}"] = state.mel_loss(x, y, fmin=fmin, fmax=fmax)
        if "stft" in selected_metrics:
            output[f"stft-{k}"] = state.stft_loss(x, y, fmin=fmin, fmax=fmax)
        if "waveform" in selected_metrics:
            output[f"waveform-{k}"] = state.waveform_loss(x, y)
        if "sisdr" in selected_metrics:
            output[f"sisdr-{k}"] = -state.sisdr_loss(x, y)
        if "gompsnr" in selected_metrics:
            output[f"gompsnr-{k}"] = state.gompsnr(
                x.audio_data.mean(dim=1),
                y.audio_data.mean(dim=1),
            ).mean()
        if "visqol" in selected_metrics:
            if k == "full":
                output[f"visqol-audio-{k}"] = metrics.quality.visqol(x, y)
            else:
                output[f"visqol-audio-{k}"] = np.nan  # placeholder if not full signal

    output["path"] = signal.path_to_file
    output.update(signal.metadata)
    return output


@argbind.bind(without_prefix=True)
@torch.no_grad()
def evaluate(
    input: str = "/N/slate/daripete/jstsp-dac/runs_32khz_bis/hb_16_1cb_4_1cb_2_1cb_wild_fr_75_320_500/objective/input",
    output: str = "/N/slate/daripete/jstsp-dac/runs_32khz_bis/hb_16_1cb_4_1cb_2_1cb_wild_fr_75_320_500/objective/output_[16, 4, 2]_23kbps",
    n_proc: int = 64,
    metrics: str = "mel,stft,waveform,sisdr,visqol,gompsnr",
):
    tracker = Tracker()
    waveform_loss = losses.L1Loss()
    stft_loss = losses.BandpassedMultiScaleSTFTLoss()
    mel_loss = losses.BandpassedMelSpectrogramLoss()
    sisdr_loss = losses.SISDRLoss()
    gompsnr = GOMPSNR(snr_type="gompsnr")
    selected_metrics = parse_metrics(metrics)

    state = State(
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        sisdr_loss=sisdr_loss,
        gompsnr=gompsnr,
        selected_metrics=selected_metrics,
    )

    audio_files = util.find_audio(input)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    @tracker.track("metrics", len(audio_files))
    def record(future, writer):
        o = future.result()
        for k, v in o.items():
            if torch.is_tensor(v):
                o[k] = v.item()
        writer.writerow(o)
        o.pop("path")
        return o

    futures = []
    metrics_suffix = "-".join(selected_metrics)
    results_csv = output.with_name(output.name + f"_metrics-{metrics_suffix}.csv")
    with tracker.live:
        with open(results_csv, "w") as csvfile:
            with ProcessPoolExecutor(
                    max_workers=n_proc,
                    mp_context=mp.get_context("spawn")
            ) as pool:
                for i in range(len(audio_files)):
                    if percent_silent(audio_files[i]) > 0.05:
                        print(f"Skipping mostly silent file: {audio_files[i]}")
                        # Skip mostly silent files
                        continue
                    print(f"Adding valid file: {audio_files[i]}")
                    future = pool.submit(
                        get_metrics, audio_files[i], output / audio_files[i].name, state, i
                    )
                    futures.append(future)
                print()

                keys = list(futures[0].result().keys())
                writer = csv.DictWriter(csvfile, fieldnames=keys)
                writer.writeheader()

                for i, future in enumerate(futures):
                    record(future, writer)

        tracker.done("test", f"N={len(audio_files)}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate()
