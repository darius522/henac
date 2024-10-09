import os
import sys
sys.path.append(os.getcwd())

import warnings
from dataclasses import dataclass
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader
from audiotools.data.datasets import ConcatDataset
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import when
from torch.utils.tensorboard import SummaryWriter

from utils.audio_utils import MultibandResampler, resample_bands

import dac

warnings.filterwarnings("ignore", category=UserWarning)

# Enable cudnn autotuner to speed up training
# (can be altered by the funcs.seed function)
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))
# Uncomment to trade memory for speed.

# Optimizers
AdamW = argbind.bind(torch.optim.AdamW, "generator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind("generator", "discriminator")
def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


# Models
DAC = argbind.bind(dac.model.DAC)
Discriminator = argbind.bind(dac.model.Discriminator)

# Data
AudioDataset = argbind.bind(AudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

# Multiband
MultibandResampler = argbind.bind(MultibandResampler)

# Loss
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(dac.nn.loss, filter_fn=filter_fn)


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform


@argbind.bind("train", "val", "test")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
):
    # Give one loader per key/value of dictionary, where
    # value is a list of folders. Create a dataset for each one.
    # Concatenate the datasets with ConcatDataset, which
    # cycles through them.
    datasets = []
    for _, v in folders.items():
        loader = AudioLoader(sources=v)
        transform = build_transform()
        dataset = AudioDataset(loader, sample_rate, transform=transform)
        datasets.append(dataset)

    dataset = ConcatDataset(datasets)
    dataset.transform = transform
    return dataset


@dataclass
class State:
    generator: DAC
    optimizer_g: AdamW
    scheduler_g: ExponentialLR

    discriminator: Discriminator
    optimizer_d: AdamW
    scheduler_d: ExponentialLR

    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: losses.L1Loss

    train_data: AudioDataset
    val_data: AudioDataset

    tracker: Tracker
    
    resampler: MultibandResampler


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: str = '',
    tag: str = "latest",
    load_weights: bool = True,
):
    generator, g_extra = None, {}
    discriminator, d_extra = None, {}

    generator = DAC() if generator is None else generator
    discriminator = Discriminator() if discriminator is None else discriminator
    if resume != "" and os.path.exists(os.path.join(resume, "dac/weights.pth")):
        generator.load_state_dict(
            torch.load(os.path.join(resume, "dac/weights.pth"), weights_only=True)[
                "state_dict"
            ],
            strict=False,
        )
        trainable_params = [
            "skip_aes",
            "multidecoders.1",
            "multidecoders.2",
        ]
        for name, param in generator.named_parameters():
            if not any([p in name for p in trainable_params]):
                print(f"Exclude parameter {name} from generator training.")
                param.requires_grad = False
            else:
                print(f"Include parameter {name} from generator training.")

    if resume != "" and os.path.exists(
        os.path.join(resume, "discriminator/weights.pth")
    ):
        discriminator.load_state_dict(torch.load(os.path.join(resume, "discriminator/weights.pth"), weights_only=True)["state_dict"],strict=False)
        trainable_params = ["discriminators.12000", "discriminators.24000"]
        for name, param in discriminator.named_parameters():
            if not any([p in name for p in trainable_params]):
                print(f"Exclude parameter {name} from discriminator training.")
                param.requires_grad = False
            else:
                print(f"Include parameter {name} from discriminator training.")
        
    resampler = MultibandResampler()

    tracker.print(generator)
    tracker.print(discriminator)

    generator = accel.prepare_model(generator, find_unused_parameters=True)
    discriminator = accel.prepare_model(discriminator)

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)

    if "optimizer.pth" in g_extra:
        optimizer_g.load_state_dict(g_extra["optimizer.pth"])
    if "scheduler.pth" in g_extra:
        scheduler_g.load_state_dict(g_extra["scheduler.pth"])
    if "tracker.pth" in g_extra:
        tracker.load_state_dict(g_extra["tracker.pth"])

    if "optimizer.pth" in d_extra:
        optimizer_d.load_state_dict(d_extra["optimizer.pth"])
    if "scheduler.pth" in d_extra:
        scheduler_d.load_state_dict(d_extra["scheduler.pth"])

    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate)

    waveform_loss = losses.L1Loss()
    stft_loss = losses.MultiScaleSTFTLoss()
    mel_loss = losses.MelSpectrogramLoss()
    gan_loss = losses.GANLoss(discriminator)

    return State(
        generator=generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=waveform_loss,
        stft_loss=stft_loss,
        mel_loss=mel_loss,
        gan_loss=gan_loss,
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
        resampler=resampler
    )


@timer()
@torch.no_grad()
def val_loop(batch, state, accel):
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)
    signals = state.val_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )
    bands = state.resampler(signals)

    # Generator output
    full_band = AudioSignal(torch.stack(bands).sum(0), signals.sample_rate)

    out = state.generator(full_band.audio_data, signals.sample_rate)
    full_recons = torch.stack(out['audio']).sum(0)
    full_recons = AudioSignal(full_recons, signals.sample_rate)

    return {
        "loss": state.mel_loss(full_recons, full_band),
        "mel/loss": state.mel_loss(full_recons, full_band),
        "stft/loss": state.stft_loss(full_recons, full_band),
        "waveform/loss": state.waveform_loss(full_recons, full_band),
    }


@timer()
def train_loop(state, batch, accel, lambdas):
    state.generator.train()
    state.discriminator.train()
    output = {}

    batch = util.prepare_batch(batch, accel.device)
    with torch.no_grad():
        signals = state.train_data.transform(
            batch["signal"].clone(), **batch["transform_args"]
        )
        bands = state.resampler(signals)

    # Generator output
    full_band = AudioSignal(torch.stack(bands).sum(0), signals.sample_rate)
    with accel.autocast():
        out = state.generator(full_band.audio_data, signals.sample_rate)
        recons_bands = out['audio']
        commitment_loss = out["vq/commitment_loss"]
        codebook_loss = out["vq/codebook_loss"]

    # Discriminator (full-band)
        for i, (recon, band, sr) in enumerate(zip(recons_bands, bands, state.resampler.cutoffs)):
            recon, band = AudioSignal(recon, signals.sample_rate), AudioSignal(band, signals.sample_rate)
            output[f"adv/disc_loss_{i}"] = output.setdefault("stft/loss", 0) + state.gan_loss.discriminator_loss(recon, band, key=str(sr))

    state.optimizer_d.zero_grad()
    accel.backward(sum([v for k, v in output.items() if 'disc_loss' in k]))
    accel.scaler.unscale_(state.optimizer_d)
    output["other/grad_norm_d"] = torch.nn.utils.clip_grad_norm_(
        state.discriminator.parameters(), 10.0
    )
    accel.step(state.optimizer_d)
    state.scheduler_d.step()

    # Generator (band-wise)
    with accel.autocast():
        for i, (recon, band, sr) in enumerate(zip(recons_bands, bands, state.resampler.cutoffs)):
            recon, band = AudioSignal(recon, signals.sample_rate), AudioSignal(band, signals.sample_rate)
            output[f"stft/loss_{i}"] = state.stft_loss(
                recon, band
            )
            output[f"mel/loss_{i}"] = state.mel_loss(
                recon, band
            )
            output[f"waveform/loss_{i}"] = state.waveform_loss(recon, band)
            (
                gen_loss,
                feat_loss,
            ) = state.gan_loss.generator_loss(recon, band, key=str(sr))
            output[f"adv/gen_loss_{i}"] = gen_loss
            output[f"adv/feat_loss_{i}"] = feat_loss

        output["vq/commitment_loss"] = commitment_loss.sum()
        output["vq/codebook_loss"] = codebook_loss.sum()
        output["loss"] = torch.tensor([0.], device='cuda')
        for k, v in lambdas.items():
            for kk, vv in output.items():
                if k in kk:
                    output["loss"] += (v * vv)

    state.optimizer_g.zero_grad()
    accel.backward(output["loss"])
    accel.scaler.unscale_(state.optimizer_g)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 1e3
    )
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()

    output["other/learning_rate"] = state.optimizer_g.param_groups[0]["lr"]
    output["other/batch_size"] = full_band.batch_size * accel.world_size

    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path):
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "mel/loss"):
        state.tracker.print(f"Best generator so far")
        tags.append("best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")

    for tag in tags:
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }
        accel.unwrap(state.generator).metadata = metadata
        accel.unwrap(state.generator).save_to_folder(
            f"{save_path}/{tag}", generator_extra
        )
        discriminator_extra = {
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }
        accel.unwrap(state.discriminator).save_to_folder(
            f"{save_path}/{tag}", discriminator_extra
        )


@torch.no_grad()
def save_samples(state, val_idx, writer):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()

    samples = [state.val_data[idx] for idx in val_idx]
    batch = state.val_data.collate(samples)
    batch = util.prepare_batch(batch, accel.device)
    signals = state.train_data.transform(
        batch["signal"].clone(), **batch["transform_args"]
    )

    bands = state.resampler(signals)
    full_band = AudioSignal(torch.stack(bands).sum(0), signals.sample_rate)
    out = state.generator(full_band.audio_data, full_band.sample_rate)
    bands_recons = torch.stack(resample_bands(out['audio'], full_band.shape[-1]))
    full_recons = AudioSignal(bands_recons.sum(0), signals.sample_rate)

    audio_dict = {"recons_full": full_recons}
    for i, band_recon in enumerate(bands_recons):
        audio_dict[f"recons_band_{i}_24khz"] = AudioSignal(band_recon, signals.sample_rate) 
    for i, (band_recon, sr) in enumerate(zip(out['audio'], [6_000, 12_000, 24_000])):
        audio_dict[f"recons_band_{i}_orig"] = AudioSignal(band_recon, sr) 

    if state.tracker.step == 0:
        audio_dict["signal_full"] = full_band
        for i, band in enumerate(bands):
            audio_dict[f"signal_band_{i}"] = AudioSignal(band, signals.sample_rate)     

    for k, v in audio_dict.items():
        for nb in range(v.batch_size):
            v[nb].cpu().write_audio_to_tb(
                f"{k}/sample_{nb}.wav", writer, state.tracker.step
            )


def validate(state, val_dataloader, accel):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel)
    # Consolidate state dicts if using ZeroRedundancyOptimizer
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()
    return output


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    lambdas: dict = {
        "mel/loss": 100.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
    },
):
    util.seed(seed)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )
    state = load(args, accel, tracker, save_path, resume=args['resume_ckpt'], load_weights=True)
    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Wrap the functions so that they neatly track in TensorBoard + progress bars
    # and only run when specific conditions are met.
    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)

    # These functions run only on the 0-rank process
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    with tracker.live:
        for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):
            train_loop(state, batch, accel, lambdas)

            last_iter = (
                tracker.step == num_iters - 1 if num_iters is not None else False
            )
            if tracker.step % sample_freq == 0 or last_iter:
                print('Saving samples ...')
                save_samples(state, val_idx, writer)

            if tracker.step % valid_freq == 0 or last_iter:
                print('Validating / Checkpointing ...')
                validate(state, val_dataloader, accel)
                checkpoint(state, save_iters, save_path)
                # Reset validation progress bar, print summary since last validation.
                tracker.done("val", f"Iteration {tracker.step}")

            if last_iter:
                break


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)
