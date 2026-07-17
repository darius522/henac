import sys, math, ast
import time

sys.path.append("/N/slate/daripete/jstsp-dac")

import dac
from audiotools import AudioSignal

import yaml
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import numpy as np
import torch
from matplotlib import pyplot as plt
import librosa

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

import argparse, os, shutil

from utils.audio_utils import normalize_to_match_peak_batched

import julius

import gc

from utils.latent_distribution import plot_band_latents, plot_codebook_indices


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


def spectrogram(signal, path):
    plt.clf()
    plt.cla()
    f = plt.figure(figsize=(10, 4))
    D = np.abs(librosa.stft(signal.audio_data.squeeze().numpy()))
    librosa.display.specshow(
        librosa.amplitude_to_db(D, ref=np.max), sr=32_000, hop_length=512, cmap="magma"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    del f
    
def match_rms(input_sig, output_sig):
    in_rms = np.sqrt(np.mean(input_sig**2))
    out_rms = np.sqrt(np.mean(output_sig**2))
    if out_rms == 0:
        return output_sig
    scale = in_rms / out_rms
    return output_sig * scale


def results_to_csv(fnames, entropies, snrs, path):
    # Convert each entry to numpy arrays and sum along the last axis
    entropies = {k: np.array(v).sum(-1) for k, v in entropies.items()}

    # Available bands (e.g., could be ["core"], ["core", "mb"], ["core", "mb", "hb"])
    bands = ["core", "mb", "hb"]
    available = [b for b in bands if b in entropies]

    # Compute total entropy as the sum of all available bands
    entropies_all = sum(entropies[b] for b in available)

    # Build the data dict dynamically (only for available bands)
    data = {"fname": fnames, "entropies_all": entropies_all}
    for b in available:
        data[f"entropies_{b}"] = entropies[b]
        data[f"snrs_{b}"] = snrs.get(b, np.zeros_like(entropies[b]))

    # Create and save the DataFrame
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def get_chunks(signal, chunk_duration, sample_rate):
    B, C, T = signal.shape
    chunk_size = int(chunk_duration * sample_rate)  # Convert duration to samples

    return [
        signal[:, :, start : min(start + chunk_size, T)]
        for start in range(0, T, chunk_size)
    ]


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


def infer_active_bands(model, audio_data, n_quantizers):
    """Run only the contiguous codec paths enabled by n_quantizers."""
    core_nq, mb_nq, hb_nq = n_quantizers
    if core_nq <= 0:
        raise ValueError("The core path must have at least one codebook.")
    if hb_nq > 0 and mb_nq <= 0:
        raise ValueError("The high-band path requires the mid-band path.")

    audio_data = model.preprocess(audio_data, sample_rate=None)
    z_core, skip_features = model.encoder(audio_data)
    z_core, core_codes, _, _, _, _ = model.quantizer(
        z_core, n_quantizers=core_nq
    )

    feats = {"core": z_core}
    codes = {"core": core_codes}
    audio = {"core": model.decode(z_core, blind_level=None)}

    if mb_nq > 0:
        mb_out = model.skip_aes[0](skip_features[1], n_quantizers=mb_nq)
        feats["mb"] = mb_out["audio"]
        codes["mb"] = mb_out["codes"]

        core_blind = model.decode(z_core, blind_level=0)
        audio["mb"] = model.multidecoders[0](
            feats["mb"], blind_level=None, x_blind=core_blind
        )

    if hb_nq > 0:
        hb_out = model.skip_aes[1](skip_features[2], n_quantizers=hb_nq)
        feats["hb"] = hb_out["audio"]
        codes["hb"] = hb_out["codes"]

        mb_blind = model.multidecoders[0](
            feats["mb"], blind_level=0, x_blind=core_blind
        )
        audio["hb"] = model.multidecoders[1](
            feats["hb"], blind_level=None, x_blind=mb_blind
        )

    return {"audio": audio, "codes": codes, "feats": feats}


def main(args):

    dataset = pd.read_csv(args.dataset)[:1000]  # Limit to 1000 samples for testing
    num_codebook = ast.literal_eval(args.num_codebooks)
    # num_codebooks = [
    #     # [[True, False], [True, False]], # 18, 0, 0
    #     # [[True, True], [True, False]], # 18, 2, 0
    #     # [[True, True], [True, True]], # 18, 2, 2
    #     # [[False, True], [True, True]], # 0, 2, 2
    #     [[True, True], [True, True]],  # 0, 0, 2
    #     # [[True, False], [False, True]], # 18, 0, 2
    # ]
    outpath = args.output_path if args.output_path is not None else args.model_path
    outpath_in = os.path.join(
        outpath, f"input"
    )  # _{num_codebook}")
    outpath_out = os.path.join(
        outpath, f"output_{num_codebook}"
    )  # _{num_codebook}")
    outpath_paths = os.path.join(
        outpath, f"paths_{num_codebook}"
    )
    os.makedirs(outpath_in, exist_ok=True)
    os.makedirs(outpath_out, exist_ok=True)
    if args.save_path_isolation:
        os.makedirs(outpath_paths, exist_ok=True)
    conf_file = "/".join(args.model_path.split("/")[:-1]) + "/conf.yaml"
    conf = get_model_args(conf_file)

    model = dac.DAC.load(
        os.path.join(args.model_path, "dac/weights.pth"), strict=True, **conf
    )
    model.eval()
    model.to("cuda")

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    # Or to count only trainable parameters:
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Example usage:
    print("Total parameters:", count_parameters(model) / 1e6)
    print("Trainable parameters:", count_trainable_parameters(model) / 1e6)

    all_feats, all_codes, snrs, fnames = (
        dict(core=[], mb=[], hb=[]),
        dict(core=[], mb=[], hb=[]),
        dict(core=[], mb=[], hb=[]),
        [],
    )
    total_infer_time_sec = 0.0
    total_audio_sec = 0.0
    infer_calls = 0
    gflops_per_infer = None
    model_warmed_up = False

    # Calculate cutoffs based on num_codebook per bands
    _, mb, hb = num_codebook

    if mb != 0 and hb != 0:
        cutoffs = [3000, 6200]
    elif mb != 0 and hb == 0:
        cutoffs = [3000]
    elif mb == 0 and hb == 0:
        cutoffs = [int(conf["sample_rate"]//2)]
    else:
        cutoffs = []

    resampler = julius.SplitBands(conf["sample_rate"], cutoffs=cutoffs).to("cuda")
    lp3_resampler = julius.SplitBands(conf["sample_rate"], cutoffs=[3000]).to("cuda")
    lp6_resampler = julius.SplitBands(conf["sample_rate"], cutoffs=[6000]).to("cuda")
    duration = 10.0
    for i, row in tqdm(dataset.iterrows(), total=len(dataset)):
        audio = AudioSignal(row.path)
        # Robustness for stereo/multi-channel inputs: force mono for DAC encoder.
        if audio.audio_data.shape[1] > 1:
            audio.audio_data = audio.audio_data.mean(dim=1, keepdim=True)
        if audio.shape[-1] < duration * conf["sample_rate"]:
            print("Audio shorter that duration, skipping!")
            continue

        signals = get_chunks(audio, duration, conf["sample_rate"])
        chunk_lengths = [s.shape[-1] for s in signals]
        signals = AudioSignal.batch(signals, pad_signals=True)
        for j, (signal, chunk_len) in enumerate(zip(signals, chunk_lengths)):
            signal.to(model.device)
            fname = os.path.basename(row.path).split(".")[0] + f"_chunk_{j}"
            fnames.append(fname)

            with torch.no_grad():
                max_codebooks = [
                    conf["n_codebooks"],
                    conf["skip_args"][0]["n_codebooks"],
                    conf["skip_args"][1]["n_codebooks"],
                ]
                if any(
                    requested < 0 or requested > maximum
                    for requested, maximum in zip(num_codebook, max_codebooks)
                ):
                    raise ValueError(
                        f"Requested codebooks {num_codebook} must be between zero "
                        f"and the configured maxima {max_codebooks}."
                    )

                if not model_warmed_up:
                    print(f"Warming up model with {args.warmup_runs} inference calls...")
                    for _ in range(args.warmup_runs):
                        infer_active_bands(model, signal.audio_data, num_codebook)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    model_warmed_up = True

                # Profile separately so profiler overhead is excluded from RTF.
                if args.profile_gflops and gflops_per_infer is None:
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if torch.cuda.is_available():
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with torch.profiler.profile(
                        activities=activities,
                        with_flops=True,
                        record_shapes=False,
                    ) as prof:
                        profile_out = infer_active_bands(
                            model, signal.audio_data, num_codebook
                        )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    total_flops = sum(
                        evt.flops for evt in prof.key_averages() if evt.flops is not None
                    )
                    gflops_per_infer = total_flops / 1e9 if total_flops > 0 else 0.0
                    del profile_out

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_t = time.perf_counter()
                out = infer_active_bands(model, signal.audio_data, num_codebook)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_t
                total_infer_time_sec += elapsed
                total_audio_sec += chunk_len / conf["sample_rate"]
                infer_calls += 1

                y_bands, codes, feats = out["audio"], out["codes"], out["feats"]
                keys = list(y_bands.keys())

                # mask for which bands to keep
                keep_mask = [n != 0 for n in num_codebook]

                # filter each dict accordingly
                y_bands = {
                    k: v
                    for k, v, keep in zip(keys, y_bands.values(), keep_mask)
                    if keep
                }
                codes = {
                    k: v for k, v, keep in zip(keys, codes.values(), keep_mask) if keep
                }
                # feats = {
                #     k: v for k, v, keep in zip(keys, feats.values(), keep_mask) if keep
                # }

            for k, c in codes.items():  # k, [B, CB, T]
                nc, fr = c.shape[1], int(c.shape[-1] // duration)
                all_codes[k].append(c.squeeze(0).detach().cpu().numpy())

            # for k, c in feats.items():  # k, [B, CB, T]
            #     all_feats[k].append(c.squeeze(0).detach().cpu().numpy())

            signal_bands = resampler(signal.audio_data)

            # Save raw decoder paths before any resampling/mixing for path-isolation analysis.
            raw_paths = {
                k: v.reshape(-1).detach().cpu().numpy()
                for k, v in y_bands.items()
            }

            mb_rec = np.zeros(signal.audio_data.shape)
            for i, ((k, y_band), signal_band) in enumerate(
                zip(y_bands.items(), signal_bands)
            ):
                y_band = resampler(y_band)[i]
                y_band = y_band.reshape(1, 1, -1).to("cuda")
                signal_band = signal_band.reshape(1, 1, -1).to("cuda")
                snrs[k].append(
                    ScaleInvariantSignalNoiseRatio()
                    .to("cuda")(y_band, signal_band)
                    .detach()
                    .cpu()
                    .item()
                )

                mb_rec = mb_rec + y_band.cpu().detach().numpy()
                del y_band

            # Save isolated and cumulative paths in waveform domain.
            if args.save_path_isolation:
                ordered_keys = [k for k in ("core", "mb", "hb") if k in raw_paths]
                running = None
                has_upper_bands = ("mb" in ordered_keys) or ("hb" in ordered_keys)
                for k in ordered_keys:
                    # Raw per-path output (before any analysis-time filtering)
                    path_sig = raw_paths[k]
                    sf.write(
                        os.path.join(outpath_paths, f"{fname}_{k}_only.wav"),
                        np.column_stack((path_sig, path_sig)),
                        samplerate=conf["sample_rate"],
                    )

                    # For cumulative analysis, enforce intended residual band assignment:
                    # - core contributes up to ~3 kHz when upper bands are present
                    # - mb contributes up to ~6 kHz when hb is present
                    path_for_cum = y_bands[k]
                    if k == "core" and has_upper_bands:
                        path_for_cum = lp3_resampler(path_for_cum)[0]
                    elif k == "mb" and ("hb" in ordered_keys):
                        path_for_cum = lp6_resampler(path_for_cum)[0]

                    path_for_cum = path_for_cum.reshape(-1).detach().cpu().numpy()
                    running = (
                        path_for_cum
                        if running is None
                        else (running + path_for_cum)
                    )
                    sf.write(
                        os.path.join(outpath_paths, f"{fname}_cum_to_{k}.wav"),
                        np.column_stack((running, running)),
                        samplerate=conf["sample_rate"],
                    )
            
            input_sig = signal.audio_data.reshape(-1).cpu().detach().numpy()
            output_sig = mb_rec.reshape(-1)
            # output_sig = match_rms(input_sig, output_sig)

            # for k, y_band in y_bands.items():
            #     sf.write(
            #         os.path.join(args.output_path, f"{fname}_{k}.wav"),
            #         y_band.reshape(-1).cpu().detach().numpy(),
            #         samplerate=32_000,
            #     )
            # sf.write(
            #     os.path.join(outpath_in, f"{fname}.wav"),
            #     np.column_stack((input_sig, input_sig)),
            #     samplerate=conf["sample_rate"],
            # )
            # sf.write(
            #     os.path.join(outpath_out, f"{fname}.wav"),
            #     np.column_stack((output_sig, output_sig)),
            #     samplerate=conf["sample_rate"],
            # )

            for k in list(codes.keys()):  # Convert to list to avoid runtime errors
                codes[k] = codes[k].detach().cpu()
            del codes

            for k in list(y_bands.keys()):
                y_bands[k] = y_bands[k].detach().cpu()
            del y_bands
            torch.cuda.empty_cache()
            gc.collect()

    # for k, v in all_feats.items():
    #     if len(v) > 0:
    #         all_feats[k] = np.concatenate(v, -1)

    # plot_codebook_indices({k: np.concatenate(v, -1) for k, v in all_codes.items()})
    # plot_band_latents(feats, projection='umap')

    bitrates = dict()
    for k, v in all_codes.items():
        if len(v) > 0:
            fr = v[0].shape[-1] // duration
            codes = np.concatenate(v, -1)
            bitrates[k] = compute_entropy(codes, N=1024, M=codes.shape[0], frame_rate=fr)

    results_to_csv(
        fnames, bitrates, snrs, path=os.path.join(args.model_path, "results.csv")
    )
    tot_ent = 0.0
    for k, bitrates in bitrates.items():
        br_per_cb = np.array(bitrates)
        print(f"Codebook Entropy for {k}: {np.round(br_per_cb, 1)}")
        print(f"Overall Entropy for {k}: {np.round(br_per_cb.sum())}")
        tot_ent += br_per_cb.sum()
    print(f"Overall Total Entropy: {np.round(tot_ent, 1)}")
    for k, v in snrs.items():
        print(f"Overall SNR for {k}: {np.round(np.mean(v), 1)}")

    if total_audio_sec > 0:
        rtf = total_infer_time_sec / total_audio_sec
        print(f"Total inference time (s): {total_infer_time_sec:.4f}")
        print(f"Total audio duration (s): {total_audio_sec:.4f}")
        print(f"Real-time factor (RTF): {rtf:.6f}")
    else:
        print("Real-time factor (RTF): N/A (no valid chunks processed)")

    if args.profile_gflops:
        if gflops_per_infer is not None:
            avg_chunk_dur = total_audio_sec / max(infer_calls, 1)
            print(f"Estimated GFLOPs per inference call: {gflops_per_infer:.4f}")
            if avg_chunk_dur > 0:
                print(
                    f"Estimated GFLOPs per audio second: {gflops_per_infer / avg_chunk_dur:.4f}"
                )
        else:
            print("Estimated GFLOPs per inference call: N/A (profiling did not run)")
        
    # lastly rename output folder to include bitrate info

    new_outpath = outpath_out + f"_{int(round(tot_ent / 1000, 1))}kbps"
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
        default="/N/slate/daripete/jstsp-dac/runs_32khz_bis/hb_16_1cb_4_1cb_2_1cb_wild_fr_75_320_500/300k",
        required=False,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/N/slate/daripete/jstsp-dac/runs_32khz_bis/hb_16_1cb_4_1cb_2_1cb_wild_fr_75_320_500/mushra/r1_deconstruction",
        required=False,
    )
    parser.add_argument('--num_codebooks', type=str, required=True, 
                    help='List of integers, e.g. "[16,4,2]"')
    parser.add_argument(
        "--save-path-isolation",
        action="store_true",
        help="Save per-path and cumulative decoder outputs for analysis.",
    )
    parser.add_argument(
        "--profile-gflops",
        action="store_true",
        help="Profile first inference call and report estimated GFLOPs.",
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
