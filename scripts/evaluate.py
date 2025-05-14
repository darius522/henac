import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools import metrics
from audiotools.core import util
from audiotools.ml.decorators import Tracker
from train import losses
from tqdm import tqdm

import julius


@dataclass
class State:
    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    waveform_loss: losses.L1Loss
    sisdr_loss: losses.SISDRLoss


def get_metrics(signal_path, recons_path, state):
    output = {}
    bands = [3000, 6000]
    signal = AudioSignal(signal_path).to("cuda")
    recons = AudioSignal(recons_path).to("cuda")
    # sp = julius.SplitBands(24_000, cutoffs=bands).to('cpu')
    # xb, yb = sp(signal.audio_data.clone()), sp(recons.audio_data.clone())
    # xb, yb = torch.concatenate([xb, signal.audio_data[None]]), torch.concatenate([yb, recons.audio_data[None]])
    # for x, y, k in zip(xb, yb, bands + [12000, 'full']):
    k = 'full'
    k = str(k)
    x = signal#AudioSignal(x, signal.sample_rate)
    y = recons#AudioSignal(y, signal.sample_rate)
    if k == 'full':
        output.update(
            {
                f"mel-{k}": state.mel_loss(x, y),
                f"stft-{k}": state.stft_loss(x, y),
                #f"waveform-{k}": state.waveform_loss(x, y),
                f"sisdr-{k}": state.sisdr_loss(x, y),
                # f"visqol-audio-{k}": metrics.quality.visqol(x, y),
                #f"visqol-speech-{k}": metrics.quality.visqol(x, y, "speech"),
            }
        )
    else:
        output.update(
            {
                f"mel-{k}": state.mel_loss(x, y),
                f"stft-{k}": state.stft_loss(x, y),
                #f"waveform-{k}": state.waveform_loss(x, y),
                f"sisdr-{k}": state.sisdr_loss(x, y),
                #f"visqol-speech-{k}": metrics.quality.visqol(x, y, "speech"),
            }
        )
    output["path"] = signal.path_to_file
    output.update(signal.metadata)
    return output


@argbind.bind(without_prefix=True)
@torch.no_grad()
def evaluate(
    input: str = "/N/slate/daripete/jstsp-dac/runs2/baseline_29cb/300k/audios/input",
    output: str = "/N/slate/daripete/jstsp-dac/runs2/baseline_29cb/300k/audios/output",
    n_proc: int = 50,
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
    
    # @tracker.track("metrics", len(audio_files))
    def record(future, writer):
        o = future#.result()
        for k, v in o.items():
            if torch.is_tensor(v):
                o[k] = v.item()
        writer.writerow(o)
        o.pop("path")
        return o

    futures = []
    # with tracker.live:
    with open(output / "metrics_all.csv", "w") as csvfile:
        for i in tqdm(range(len(audio_files))):
            future = get_metrics(audio_files[i], output / audio_files[i].name, state)
            futures.append(future)

        keys = list(futures[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()

        for future in futures:
            record(future, writer)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate()
