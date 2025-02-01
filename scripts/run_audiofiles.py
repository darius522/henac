import sys

sys.path.append('/N/slate/daripete/jstsp-dac')

import dac
from audiotools import AudioSignal

import yaml
import pandas as pd
from tqdm import tqdm
import soundfile as sf 
import numpy as np

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

import argparse, os

from utils.audio_utils import normalize_to_match_peak_batched

import julius


def compute_entropy(code_tensor, N=1024, M=4, frame_rate=500):
    """
    Calculate the entropy-based bitrate of an RVQ-based neural audio codec.
    
    Args:
    - code_tensor (ndarray): The code tensor of shape [M, T] where M is the number of codebooks and T is the number of time frames.
    - N (int): The size of each codebook (number of entries in the codebook).
    - M (int): The number of codebooks.
    - frame_rate (int): The frame rate (frames per second).
    
    Returns:
    - bitrate (float): The entropy-based bitrate in bits per second.
    """
    import numpy as np
    from scipy.stats import entropy
    # Initialize the total entropy
    total_entropy, total_bitrate = [], []
    
    # Loop over each codebook (axis 0)
    for i in range(M):
        # Get the codebook indices for this codebook (shape [T,])
        codebook_indices = code_tensor[i]
        
        # Compute the frequency distribution of the indices in the codebook
        # Calculate probabilities (relative frequencies)
        counts = np.bincount(codebook_indices, minlength=N)
        probabilities = counts / len(codebook_indices)
        
        # Calculate the entropy for this codebook
        codebook_entropy = entropy(probabilities, base=2)  # Entropy in bits
        
        # Add the entropy for this codebook to the total entropy
        total_entropy.append(codebook_entropy)
        total_bitrate.append(codebook_entropy*frame_rate)
    
    return total_bitrate

def get_model_args(conf_path):
    with open(conf_path, "r") as file:
        config = yaml.safe_load(file)

    # Extract DAC-specific arguments
    dac_prefix = "DAC."
    return {
        key[len(dac_prefix):]: value for key, value in config.items() if key.startswith(dac_prefix)
    }


def main(args):

    outpath = os.path.join(args.model_path, 'audios')
    os.makedirs(outpath, exist_ok=True)
    conf_file = '/'.join(args.model_path.split('/')[:-1]) + '/conf.yaml'
    
    import pdb; pdb.set_trace()
    
    model = dac.DAC.load(os.path.join(args.model_path, 'dac/weights.pth'), strict=True, kwargs=get_model_args(conf_file))
    model.eval()
    model.to("cuda")

    dataset = pd.read_csv(args.dataset)
    entropies, snrs = dict(core=[], mb=[]), dict(band=[], full=[])
    
    resampler = julius.SplitBands(24_000, cutoffs=[3000, 6000]).to('cuda')
    duration = 5.0
    num_codebook = [24, 2]
    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        fname = os.path.basename(row.path).split('.')[0]
        # Load audio signal file
        signal = AudioSignal(row.path, duration=duration)

        signal.to(model.device)

        out = model.forward(signal.audio_data, n_quantizers=num_codebook)
        y_blind = model.decode(out['z'])
        y_band, mb_codes, core_codes = out['audio'], out['codes'], out['core_codes']
        entropies['core'].append(compute_entropy(core_codes.squeeze(0).detach().cpu().numpy(), M=num_codebook[0], frame_rate=75))
        entropies['mb'].append(compute_entropy(mb_codes.squeeze(0).detach().cpu().numpy(), M=num_codebook[1]))
        
        signal_band = resampler(signal.audio_data)[1:].sum(0)
        y_band = y_band.reshape(1, 1, -1).to('cuda')
        signal_band = signal_band.reshape(1, 1, -1).to('cuda')
        
        # also reconstruct the multiband signal
        mb_rec = (resampler(y_blind)[0] + y_band).to('cpu').detach()

        y_blind, signal = y_blind.to('cpu').detach(), signal.audio_data.to('cpu').detach()
        y_band, signal_band = y_band.to('cpu').detach(), signal_band.to('cpu').detach()
        snrs['full'].append(ScaleInvariantSignalNoiseRatio().to("cpu")(y_blind, signal))
        snrs['band'].append(ScaleInvariantSignalNoiseRatio().to("cpu")(y_band, signal_band))
        
        
        sf.write(
            os.path.join(outpath, f"{fname}_full_input.wav"),
            signal.reshape(-1).numpy(),
            samplerate=24_000,
        )
        sf.write(
            os.path.join(outpath, f"{fname}_full_output.wav"),
            y_blind.reshape(-1).numpy(),
            samplerate=24_000,
        )
        # sf.write(
        #     os.path.join(outpath, f"{fname}_band_input.wav"),
        #     signal_band.reshape(-1).numpy(),
        #     samplerate=24_000,
        # )
        # sf.write(
        #     os.path.join(outpath, f"{fname}_band_output.wav"),
        #     y_band.reshape(-1).numpy(),
        #     samplerate=24_000,
        # )
        sf.write(
            os.path.join(outpath, f"{fname}_mb_rec.wav"),
            mb_rec.reshape(-1).numpy(),
            samplerate=24_000,
        )
        
        del out
    
    for k, entropies in entropies.items():
        br_per_cb = np.array(entropies).mean(0)
        print(f'Codebook Entropy for {k}: {np.round(br_per_cb, 1)}')
        print(f'Overall Entropy for {k}: {np.round(br_per_cb.sum(), 1)}')
    for k, v in snrs.items():
        print(f'Overall SNR for {k}: {np.round(np.mean(v), 1)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run compression on a dataset. Save the audio files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/N/slate/daripete/jstsp-dac/datasets/fma_test_subset.csv",
        required=False,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/N/slate/daripete/jstsp-dac/runs/highband_cb1_8/latest",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        required=False,
    )
    args = parser.parse_args()

    main(args)
