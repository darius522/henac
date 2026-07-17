import sys, math, ast
import time
import yaml

sys.path.append("/N/slate/daripete/jstsp-dac")

import dac
from audiotools import AudioSignal
from dac.nn.quantize import VectorQuantize

import pandas as pd
import torch
from tqdm import tqdm
import soundfile as sf
import numpy as np

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

import argparse, os, shutil

from utils.audio_utils import normalize_to_match_peak_batched

import julius

from matplotlib import pyplot as plt


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
        total_bitrate.append(codebook_entropy * frame_rate)

    return total_bitrate


def results_to_csv(fnames, entropies, snrs, path):
    data = {
        "fname": fnames,
        "entropies": entropies,
        "snrs": snrs,
    }

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def get_chunks(signal, chunk_duration, sample_rate):
    B, C, T = signal.shape
    chunk_size = int(chunk_duration * sample_rate)  # Convert duration to samples

    return [
        signal[:, :, start : min(start + chunk_size, T)]
        for start in range(0, T, chunk_size)
    ]
    
def match_rms(input_sig, output_sig):
    in_rms = np.sqrt(np.mean(input_sig**2))
    out_rms = np.sqrt(np.mean(output_sig**2))
    if out_rms == 0:
        return output_sig
    scale = in_rms / out_rms
    return output_sig * scale

def get_model_args(conf_path):
    with open(conf_path, "r") as file:
        config = yaml.safe_load(file)

    # Extract DAC-specific arguments
    dac_prefix = "DAC."
    return {
        key[len(dac_prefix) :]: value
        for key, value in config.items()
        if key.startswith(dac_prefix)
    }


def profile_model_compute(model, audio_data, n_quantizers):
    """Count MACs for executed conv/linear layers and RVQ distance searches.

    One multiply-accumulate (MAC) is reported as two FLOPs. Bias, activation,
    normalization, indexing, and other elementwise operations are omitted.
    """
    macs = {
        "conv1d": 0,
        "conv_transpose1d": 0,
        "linear": 0,
        "vq_distance": 0,
    }
    handles = []

    def convolution_hook(module, inputs, output):
        input_tensor = inputs[0]
        kernel_size = module.kernel_size[0]
        if isinstance(module, torch.nn.ConvTranspose1d):
            # Each input value contributes one kernel per output channel.
            macs["conv_transpose1d"] += (
                input_tensor.shape[0]
                * input_tensor.shape[1]
                * input_tensor.shape[-1]
                * (module.out_channels // module.groups)
                * kernel_size
            )
        else:
            # Each output value computes a kernel-sized dot product.
            macs["conv1d"] += (
                output.numel()
                * (module.in_channels // module.groups)
                * kernel_size
            )

    def linear_hook(module, inputs, output):
        macs["linear"] += output.numel() * module.in_features

    def vector_quantize_hook(module, inputs, output):
        latent = inputs[0]
        batch_size, _, num_frames = latent.shape
        macs["vq_distance"] += (
            batch_size
            * num_frames
            * module.codebook_dim
            * module.codebook_size
        )

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            handles.append(module.register_forward_hook(convolution_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, VectorQuantize):
            handles.append(module.register_forward_hook(vector_quantize_hook))

    try:
        with torch.no_grad():
            output = model.forward(audio_data, n_quantizers=n_quantizers)
    finally:
        for handle in handles:
            handle.remove()

    return output, macs


def main(args):

    num_codebook = ast.literal_eval(args.num_codebooks)

    dataset = pd.read_csv(args.dataset)[:1000]
    all_codes, snrs, fnames = [], [], []

    outpath = args.output_path if args.output_path is not None else args.model_path
    outpath_in = os.path.join(outpath, "input")
    outpath_out = os.path.join(
        outpath, f"output_{num_codebook}"
    )
    os.makedirs(outpath_in, exist_ok=True)
    os.makedirs(outpath_out, exist_ok=True)
    
    conf_file = "/".join(args.model_path.split("/")[:-1]) + "/conf.yaml"

    conf_dict = get_model_args(conf_file)
    model = dac.DAC.load(
        os.path.join(args.model_path, "dac/weights.pth"),
        strict=True,
        **conf_dict,
    )
    model.eval()
    model.to("cuda")

    duration = 10.0
    total_infer_time_sec = 0.0
    total_audio_sec = 0.0
    infer_calls = 0
    gflops_per_infer = None
    gmacs_per_infer = None
    mac_breakdown = None
    profiled_audio_sec = None
    model_warmed_up = False

    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        audio = AudioSignal(row.path)
        # The DAC encoder expects mono input shaped [B, 1, T].
        if audio.audio_data.shape[1] > 1:
            audio.audio_data = audio.audio_data.mean(dim=1, keepdim=True)
        if audio.shape[-1] < duration * conf_dict["sample_rate"]:
            print("Audio shorter that duration, skipping!")
            continue

        signals = get_chunks(audio, duration, conf_dict["sample_rate"])
        signals = AudioSignal.batch(signals, pad_signals=True)
        for j, signal in enumerate(signals):
            try:
                signal.to(model.device)
                fname = os.path.basename(row.path).split(".")[0] + f"_chunk_{j}"
                fnames.append(fname)

                with torch.no_grad():
                    if not model_warmed_up:
                        print(
                            f"Warming up model with {args.warmup_runs} "
                            "inference calls..."
                        )
                        for _ in range(args.warmup_runs):
                            warmup_out = model.forward(
                                signal.audio_data, n_quantizers=num_codebook
                            )
                        if args.warmup_runs > 0:
                            del warmup_out
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        model_warmed_up = True

                    # Count operations separately so hook overhead is excluded from RTF.
                    if args.profile_gflops and gflops_per_infer is None:
                        profile_out, mac_breakdown = profile_model_compute(
                            model, signal.audio_data, num_codebook
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        total_macs = sum(mac_breakdown.values())
                        gmacs_per_infer = total_macs / 1e9
                        gflops_per_infer = 2 * gmacs_per_infer
                        profiled_audio_sec = (
                            signal.audio_data.shape[-1] / conf_dict["sample_rate"]
                        )
                        del profile_out

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_t = time.perf_counter()
                    out = model.forward(
                        signal.audio_data, n_quantizers=num_codebook
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    total_infer_time_sec += time.perf_counter() - start_t
                    total_audio_sec += (
                        signal.audio_data.shape[-1] / conf_dict["sample_rate"]
                    )
                    infer_calls += 1

                y, codes = out["audio"], out["codes"]
                # bands = resampler(signal.audio_data)
                all_codes.append(codes.squeeze(0).detach().cpu().numpy())

                y, signal = y.to("cpu").detach(), signal.audio_data.to("cpu").detach()
                # y = y / torch.max(torch.abs(y)) * 0.99  # prevent clipping
                snrs.append(ScaleInvariantSignalNoiseRatio().to("cpu")(y, signal))
                
                input_sig = signal.reshape(-1).numpy()
                output_sig = y.reshape(-1).numpy()
                # output_sig = match_rms(input_sig, output_sig)

                # sf.write(
                #     os.path.join(outpath_in, f"{fname}.wav"),
                #     np.column_stack((input_sig, input_sig)),
                #     samplerate=conf_dict["sample_rate"],
                # )
                sf.write(
                    os.path.join(outpath_out, f"{fname}.wav"),
                    np.column_stack((output_sig, output_sig)),
                    samplerate=conf_dict["sample_rate"],
                )

                del out
            except Exception as e:
                print(f"Error processing {row.path} chunk {j}: {e}")
                continue

    if not all_codes:
        raise RuntimeError(
            "No audio chunks were processed successfully; no results can be computed."
        )

    fr = all_codes[0].shape[-1] // duration
    codes = np.concatenate(all_codes, -1)
    bitrates = compute_entropy(codes, N=1024, M=codes.shape[0], frame_rate=fr)
    results_to_csv(
        fnames,
        np.array(bitrates).sum(),  # sum over cb
        np.array(snrs),
        path=os.path.join(args.model_path, "results.csv"),
    )
    br_per_cb = np.array(bitrates)
    print(f"Codebook Entropy: {np.round(br_per_cb, 1)}")
    print(f"Overall Entropy: {np.round(br_per_cb.sum(), 1)}")
    print(f"Overall SNR: {np.round(np.mean(snrs), 1)}")

    if total_audio_sec > 0:
        rtf = total_infer_time_sec / total_audio_sec
        print(f"Total inference calls: {infer_calls}")
        print(f"Total inference time (s): {total_infer_time_sec:.4f}")
        print(f"Total processed audio duration (s): {total_audio_sec:.4f}")
        print(f"Real-time factor (RTF): {rtf:.6f}")
    else:
        print("Real-time factor (RTF): N/A (no valid chunks processed)")

    if args.profile_gflops:
        if gflops_per_infer is not None:
            print("Compute convention: 1 MAC = 2 FLOPs")
            for operation, operation_macs in mac_breakdown.items():
                print(f"  {operation} GMACs per call: {operation_macs / 1e9:.4f}")
            print(f"Estimated GMACs per inference call: {gmacs_per_infer:.4f}")
            print(f"Estimated GFLOPs per inference call: {gflops_per_infer:.4f}")
            if profiled_audio_sec and profiled_audio_sec > 0:
                print(
                    "Estimated GMACs per audio second: "
                    f"{gmacs_per_infer / profiled_audio_sec:.4f}"
                )
                print(
                    "Estimated GFLOPs per audio second: "
                    f"{gflops_per_infer / profiled_audio_sec:.4f}"
                )
        else:
            print("Estimated GFLOPs per inference call: N/A (profiling did not run)")

    new_outpath = outpath_out + f"_{int(round(br_per_cb.sum() / 1000, 0))}kbps"
    # If destination exists, delete it
    if os.path.exists(new_outpath):
        shutil.rmtree(new_outpath)
    os.rename(outpath_out, new_outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run compression on a dataset. Save the audio files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="/N/slate/daripete/jstsp-dac/runs_32khz_bis/hb_16_1cb_4_1cb_2_1cb_wild_fr_75_320_500/mushra/input/wav_paths.csv",
        required=False,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/N/slate/daripete/jstsp-dac/runs_32khz/baseline_32_1cb_large_fr_80/300k",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/N/slate/daripete/jstsp-dac/__tmp",
        required=False,
    )
    parser.add_argument('--num_codebooks', type=str, required=True, 
                    help='List of integers, e.g. "[32, 1]"')
    parser.add_argument(
        "--profile-gflops",
        action="store_true",
        help="Profile the first inference call and report estimated GFLOPs.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Number of untimed GPU inference calls before benchmarking (default: 3).",
    )
    args = parser.parse_args()

    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be non-negative")

    main(args)
