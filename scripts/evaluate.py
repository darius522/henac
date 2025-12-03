import csv
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import argbind
import torch, torchaudio
from audiotools import AudioSignal
from audiotools import metrics
from audiotools.core import util
from audiotools.ml.decorators import Tracker
from train import losses

import julius


@dataclass
class State:
    stft_loss: losses.BandpassedMultiScaleSTFTLoss
    mel_loss: losses.BandpassedMelSpectrogramLoss
    waveform_loss: losses.L1Loss
    sisdr_loss: losses.SISDRLoss

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
        output.update(
            {
                f"mel-{k}": state.mel_loss(x, y, fmin=fmin, fmax=fmax),
                f"stft-{k}": state.stft_loss(x, y, fmin=fmin, fmax=fmax),
                f"waveform-{k}": state.waveform_loss(x, y),
                f"sisdr-{k}": -state.sisdr_loss(x, y),
            }
        )
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
    input: str = "/N/slate/daripete/jstsp-dac/runs_32khz/hb_16cb_4_1cb_2_1cb_fr_80_320_500/300k/audios/input",
    output: str = "/N/slate/daripete/jstsp-dac/runs_32khz/hb_16cb_4_1cb_2_1cb_fr_80_320_500/300k/audios/output_[16, 1, 0]_14kbps",
    n_proc: int = 64,
):
    tracker = Tracker()
    waveform_loss = losses.L1Loss()
    stft_loss = losses.BandpassedMultiScaleSTFTLoss()
    mel_loss = losses.BandpassedMelSpectrogramLoss()
    sisdr_loss = losses.SISDRLoss()

    state = State(
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        sisdr_loss=sisdr_loss,
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
    with tracker.live:
        with open(output.with_name(output.name + ".csv"), "w") as csvfile:
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
