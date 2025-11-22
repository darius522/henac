import csv
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools import metrics
from audiotools.core import util
from audiotools.ml.decorators import Tracker
from train import losses

import julius


@dataclass
class State:
    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    waveform_loss: losses.L1Loss
    sisdr_loss: losses.SISDRLoss


def get_metrics(signal_path, recons_path, state, idx):
    output = {}
    bands = [3000, 6000]
    signal = AudioSignal(signal_path)
    recons = AudioSignal(recons_path)
    sp = julius.SplitBands(32_000, cutoffs=bands)
    xb = torch.concat([sp(signal.audio_data.clone()), torch.sum(sp(signal.audio_data.clone()), 0, keepdim=True)], 0)
    yb = torch.concat([sp(recons.audio_data.clone()), torch.sum(sp(recons.audio_data.clone()), 0, keepdim=True)], 0)
    for x, y, k in zip(xb, yb, bands + [16000, 'full']):
        k = str(k)
        x = AudioSignal(x, signal.sample_rate)
        y = AudioSignal(y, signal.sample_rate)

        output.update(
            {
                f"mel-{k}": state.mel_loss(x, y),
                f"stft-{k}": state.stft_loss(x, y),
                f"waveform-{k}": state.waveform_loss(x, y),
                f"sisdr-{k}": -state.sisdr_loss(x, y),
            }
        )
        if k == "full":
            output[f"visqol-audio-{k}"] = metrics.quality.visqol(x, y)
        else:
            output[f"visqol-audio-{k}"] = np.nan  # placeholder

    output["path"] = signal.path_to_file
    output.update(signal.metadata)
    print(f'proc: {idx}')
    return output


@argbind.bind(without_prefix=True)
@torch.no_grad()
def evaluate(
    input: str = "/N/slate/daripete/jstsp-dac/runs_32khz/hb_16cb_4_1cb_2_1cb_fr_80_320_500/300k/audios/input",
    output: str = "/N/slate/daripete/jstsp-dac/runs_32khz/hb_16cb_4_1cb_2_1cb_fr_80_320_500/300k/audios/output_[16, 1, 0]_14kbps",
    n_proc: int = 32,
):
    tracker = Tracker()
    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
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
            with ProcessPoolExecutor(n_proc, mp.get_context("fork")) as pool:
                for i in range(len(audio_files)):
                    future = pool.submit(
                        get_metrics, audio_files[i], output / audio_files[i].name, state, i
                    )
                    futures.append(future)

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