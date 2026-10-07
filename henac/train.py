"""Train HENAC's core, mid, or high path with the same training loop."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List

import argbind
import julius
import torch
import yaml
from audiotools import AudioSignal, ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.ml.decorators import Tracker
from torch.utils.tensorboard import SummaryWriter

from henac import model, nn
from henac.data import RandomAudioDataset
from henac.training_stage import configure_trainable, initialize_from_previous, stage_target


torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", "1")))

Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)
DAC = argbind.bind(model.DAC)
Discriminator = argbind.bind(model.Discriminator)

tfm = SimpleNamespace(
    **{
        name: argbind.bind(getattr(transforms, name), "train", "val")
        for name in ("Identity", "VolumeNorm", "RescaleAudio", "ShiftPhase")
    }
)
losses = SimpleNamespace(
    **{
        name: argbind.bind(getattr(nn.loss, name))
        for name in ("L1Loss", "MultiScaleSTFTLoss", "MelSpectrogramLoss", "GANLoss")
    }
)


@argbind.bind("generator", "discriminator")
def AdamW(
    parameters, lr: float = 1e-4, betas: List[float] = [0.8, 0.99], weight_decay: float = 0.01
):
    return torch.optim.AdamW(parameters, lr=lr, betas=tuple(betas), weight_decay=weight_decay)


@argbind.bind("generator", "discriminator")
def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["VolumeNorm", "RescaleAudio", "ShiftPhase"],
):
    def compose(names, name, prob=1.0):
        return transforms.Compose(*(getattr(tfm, item)() for item in names), name=name, prob=prob)

    return transforms.Compose(
        compose(preprocess, "preprocess"),
        compose(augment, "augment", augment_prob),
        compose(postprocess, "postprocess"),
    )


@argbind.bind("train", "val")
def build_dataset(
    sample_rate: int,
    folders: dict = None,
    source: str = "",
    duration: float = 0.38,
    n_examples: int = 10000000,
    seed: int = 0,
):
    """Use a single path or the original balanced groups of audio sources."""
    if source:
        folders = {"audio": [source]}
    if not folders:
        raise ValueError("Provide train_path/val_path or build_dataset.folders")
    transform = build_transform()
    return RandomAudioDataset(
        folders, sample_rate, duration, n_examples, transform=transform, seed=seed
    )


@dataclass
class State:
    generator: torch.nn.Module
    discriminator: torch.nn.Module
    optimizer_g: torch.optim.Optimizer
    optimizer_d: torch.optim.Optimizer
    scheduler_g: torch.optim.lr_scheduler.LRScheduler
    scheduler_d: torch.optim.lr_scheduler.LRScheduler
    stft_loss: torch.nn.Module
    mel_loss: torch.nn.Module
    gan_loss: torch.nn.Module
    waveform_loss: torch.nn.Module
    train_data: RandomAudioDataset
    val_data: RandomAudioDataset
    splitter: julius.SplitBands
    tracker: Tracker
    stage: str
    scaler: float
    next_step: int
    best_mel: float


def _load_checkpoint(path: str) -> dict:
    checkpoint = Path(path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "dac/weights.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return torch.load(checkpoint, map_location="cpu", weights_only=True)


def _model_state(checkpoint: dict) -> dict:
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    raise ValueError("Checkpoint has no model_state or state_dict")


def build_state(
    args,
    accel,
    tracker,
    init_from,
    resume_from,
    train_path,
    val_path,
    decoder_frozen,
    band_scaler,
    band_cutoffs,
    seed,
):
    generator = DAC()
    discriminator = Discriminator()
    stage = generator.stage
    if init_from and resume_from:
        raise ValueError("init_from and resume_from are mutually exclusive")
    if stage == "core" and init_from:
        raise ValueError("The core stage starts from scratch; use resume_from to continue it")
    if stage != "core" and not (init_from or resume_from):
        raise ValueError(f"The {stage} stage requires init_from or resume_from")

    resume_state = None
    if init_from:
        previous = _load_checkpoint(init_from)
        expected_stage = "core" if stage == "mid" else "mid"
        if previous.get("stage", expected_stage) != expected_stage:
            raise ValueError(f"{stage} must initialize from a {expected_stage} checkpoint")
        initialize_from_previous(generator, _model_state(previous))
    if resume_from:
        resume_state = _load_checkpoint(resume_from)
        if "optimizer_g" not in resume_state:
            raise ValueError("resume_from requires a complete HENAC training checkpoint")
        if resume_state["stage"] != stage:
            raise ValueError("The checkpoint stage differs from DAC.stage")
        model_config = {key[4:]: value for key, value in args.items() if key.startswith("DAC.")}
        if resume_state["model_config"] != model_config:
            raise ValueError("Resume model configuration differs from the checkpoint")
        generator.load_state_dict(_model_state(resume_state), strict=True)
        discriminator.load_state_dict(resume_state["discriminator_state"], strict=True)

    active = configure_trainable(generator, decoder_frozen)
    tracker.print(
        f"Stage {stage}: {len(active)} trainable tensors, "
        f"{sum(p.numel() for p in generator.parameters() if p.requires_grad):,} trainable parameters"
    )

    model_kwargs = {"find_unused_parameters": True} if accel.use_ddp else {}
    generator = accel.prepare_model(generator, **model_kwargs)
    discriminator = accel.prepare_model(discriminator, **model_kwargs)
    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(p for p in generator.parameters() if p.requires_grad)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters())
        scheduler_d = ExponentialLR(optimizer_d)

    next_step, best_mel = 0, float("inf")
    if resume_state is not None:
        optimizer_g.load_state_dict(resume_state["optimizer_g"])
        optimizer_d.load_state_dict(resume_state["optimizer_d"])
        scheduler_g.load_state_dict(resume_state["scheduler_g"])
        scheduler_d.load_state_dict(resume_state["scheduler_d"])
        next_step = int(resume_state["next_step"])
        best_mel = float(resume_state["best_mel"])
        tracker.step = next_step
        tracker.print(f"Resuming {stage} at optimizer step {next_step}")

    sample_rate = accel.unwrap(generator).sample_rate
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rate, source=train_path, seed=seed)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rate, source=val_path, seed=seed + 1)
    if len(band_cutoffs) != 2 or not 0 < band_cutoffs[0] < band_cutoffs[1] < sample_rate / 2:
        raise ValueError("band_cutoffs must contain two increasing frequencies below Nyquist")
    splitter = julius.SplitBands(sample_rate, cutoffs=band_cutoffs).to(accel.device)
    return State(
        generator=generator,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        stft_loss=losses.MultiScaleSTFTLoss(),
        mel_loss=losses.MelSpectrogramLoss(),
        gan_loss=losses.GANLoss(discriminator),
        waveform_loss=losses.L1Loss(),
        train_data=train_data,
        val_data=val_data,
        splitter=splitter,
        tracker=tracker,
        stage=stage,
        scaler=band_scaler,
        next_step=next_step,
        best_mel=best_mel,
    )


def _prepare_signal(batch, dataset, accel):
    batch = util.prepare_batch(batch, accel.device)
    with torch.no_grad():
        return dataset.transform(batch["signal"].clone(), **batch["transform_args"])


def train_step(state, batch, accel, lambdas):
    state.generator.train()
    state.discriminator.train()
    signal = _prepare_signal(batch, state.train_data, accel)

    with accel.autocast():
        output = state.generator(signal.audio_data, signal.sample_rate, step=state.tracker.step)
        target_data = stage_target(signal.audio_data, state.stage, state.splitter, state.scaler)
        target = AudioSignal(target_data, signal.sample_rate)
        reconstruction = AudioSignal(output["audio"], signal.sample_rate)
        discriminator_loss = state.gan_loss.discriminator_loss(reconstruction, target)

    state.optimizer_d.zero_grad(set_to_none=True)
    accel.backward(discriminator_loss)
    accel.scaler.unscale_(state.optimizer_d)
    grad_d = torch.nn.utils.clip_grad_norm_(state.discriminator.parameters(), 10.0)
    accel.step(state.optimizer_d)
    state.scheduler_d.step()

    with accel.autocast():
        metrics = {
            "stft/loss": state.stft_loss(reconstruction, target),
            "mel/loss": state.mel_loss(reconstruction, target),
            "waveform/loss": state.waveform_loss(reconstruction, target),
            "vq/commitment_loss": output["vq/commitment_loss"],
            "vq/codebook_loss": output["vq/codebook_loss"],
            "vq/entropy_loss": output["vq/entropy_loss"],
        }
        metrics["adv/gen_loss"], metrics["adv/feat_loss"] = state.gan_loss.generator_loss(
            reconstruction, target
        )
        total = sum(weight * metrics[name] for name, weight in lambdas.items())

    state.optimizer_g.zero_grad(set_to_none=True)
    accel.backward(total)
    accel.scaler.unscale_(state.optimizer_g)
    grad_g = torch.nn.utils.clip_grad_norm_(state.generator.parameters(), 1e3)
    accel.step(state.optimizer_g)
    state.scheduler_g.step()
    accel.update()
    metrics.update(
        {
            "loss": total,
            "adv/disc_loss": discriminator_loss,
            "other/grad_norm_g": grad_g,
            "other/grad_norm_d": grad_d,
        }
    )
    return {name: float(value.detach()) for name, value in metrics.items()}


@torch.no_grad()
def validate(state, dataloader, accel):
    state.generator.eval()
    totals = torch.zeros(2, device=accel.device)
    for batch in dataloader:
        signal = _prepare_signal(batch, state.val_data, accel)
        target_data = stage_target(signal.audio_data, state.stage, state.splitter, state.scaler)
        target = AudioSignal(target_data, signal.sample_rate)
        output = state.generator(signal.audio_data, signal.sample_rate)
        reconstruction = AudioSignal(output["audio"], signal.sample_rate)
        totals[0] += state.mel_loss(reconstruction, target).detach()
        totals[1] += 1
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals)
    return float((totals[0] / totals[1]).item())


def save_checkpoint(state, accel, save_path: Path, next_step: int, tag: str, model_config: dict):
    """Atomically replace one complete, resumable checkpoint file."""
    checkpoint = {
        "format_version": 1,
        "stage": state.stage,
        "next_step": next_step,
        "best_mel": state.best_mel,
        "model_config": model_config,
        "model_state": accel.unwrap(state.generator).state_dict(),
        "discriminator_state": accel.unwrap(state.discriminator).state_dict(),
        "optimizer_g": state.optimizer_g.state_dict(),
        "optimizer_d": state.optimizer_d.state_dict(),
        "scheduler_g": state.scheduler_g.state_dict(),
        "scheduler_d": state.scheduler_d.state_dict(),
    }
    destination = save_path / f"{tag}.pth"
    temporary = save_path / f"{tag}.pth.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)


def save_best_model(state, accel, save_path: Path, model_config: dict):
    destination = save_path / "best.pth"
    temporary = save_path / "best.pth.tmp"
    torch.save(
        {
            "format_version": 1,
            "stage": state.stage,
            "model_config": model_config,
            "model_state": accel.unwrap(state.generator).state_dict(),
        },
        temporary,
    )
    os.replace(temporary, destination)


def infinite_batches(dataloader):
    while True:
        yield from dataloader


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "runs/henac",
    init_from: str = "",
    resume_from: str = "",
    train_path: str = "",
    val_path: str = "",
    decoder_frozen: bool = False,
    band_scaler: float = 1.0,
    band_cutoffs: list = [3000, 6000],
    num_iters: int = 400000,
    save_iters: list = [50000, 100000, 200000, 300000, 400000],
    valid_freq: int = 1000,
    log_freq: int = 100,
    batch_size: int = 8,
    val_batch_size: int = 1,
    num_workers: int = 8,
    lambdas: dict = {
        "mel/loss": 15.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
    },
):
    if lambdas.get("vq/entropy_loss", 0.0):
        raise ValueError("The reference training has no explicit entropy loss")
    if valid_freq <= 0 or log_freq <= 0:
        raise ValueError("valid_freq and log_freq must be positive")
    util.seed(seed)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_path / "logs")) if accel.local_rank == 0 else None
    tracker = Tracker(writer=writer, log_file=str(save_path / "log.txt"), rank=accel.local_rank)

    state = build_state(
        args,
        accel,
        tracker,
        init_from,
        resume_from,
        train_path,
        val_path,
        decoder_frozen,
        band_scaler,
        band_cutoffs,
        seed,
    )
    if accel.local_rank == 0:
        with (save_path / "conf.yaml").open("w") as file:
            yaml.safe_dump(dict(args), file, sort_keys=False)
    model_config = {key[4:]: value for key, value in args.items() if key.startswith("DAC.")}
    train_loader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.next_step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    val_loader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
    )
    batches = infinite_batches(train_loader)
    with tracker.live:
        for step in range(state.next_step, num_iters):
            tracker.step = step
            metrics = train_step(state, next(batches), accel, lambdas)
            next_step = step + 1
            if accel.local_rank == 0 and (next_step % log_freq == 0 or next_step == 1):
                tracker.print(
                    f"step {next_step}/{num_iters}: "
                    f"loss={metrics['loss']:.4f}, mel={metrics['mel/loss']:.4f}"
                )
                if writer is not None:
                    for name, value in metrics.items():
                        writer.add_scalar(f"train/{name}", value, next_step)

            if next_step % valid_freq == 0 or next_step == num_iters:
                mel = validate(state, val_loader, accel)
                if accel.local_rank == 0:
                    tracker.print(f"validation step {next_step}: mel={mel:.4f}")
                    if writer is not None:
                        writer.add_scalar("val/mel_loss", mel, next_step)
                    improved = mel < state.best_mel
                    state.best_mel = min(mel, state.best_mel)
                    save_checkpoint(state, accel, save_path, next_step, "latest", model_config)
                    if improved:
                        save_best_model(state, accel, save_path, model_config)
                    if next_step in save_iters:
                        shutil.copyfile(
                            save_path / "latest.pth", save_path / f"{next_step // 1000}k.pth"
                        )
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
    if writer is not None:
        writer.close()


def main():
    args = argbind.parse_args()
    args["args.debug"] = bool(int(os.getenv("HENAC_DEBUG", "0")))
    try:
        with argbind.scope(args):
            with Accelerator() as accelerator:
                train(args, accelerator)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
