"""Trainable parameters and checkpoint transitions for HENAC's three stages."""

from collections.abc import Mapping

import torch

from henac.model import DAC


STAGES = ("core", "mid", "high")


def configure_trainable(model: DAC, decoder_frozen: bool = False) -> list[str]:
    """Freeze earlier paths and return the names optimized in this stage."""
    stage = model.stage
    if stage not in STAGES:
        raise ValueError(f"Unknown training stage: {stage}")
    active = []
    for name, parameter in model.named_parameters():
        if stage == "core":
            trainable = name.startswith(("encoder.", "quantizer.", "decoder.model.0.")) or (
                name.startswith("multidecoders.0.") and ".t_conv.skip." not in name
            )
        else:
            index = 0 if stage == "mid" else 1
            skip_prefix = f"skip_aes.{index}."
            decoder_prefix = f"multidecoders.{index}."
            skip_merger = f"{decoder_prefix}model.0.t_conv.skip."
            if decoder_frozen:
                decoder_trainable = name.startswith(skip_merger)
            else:
                blind_prefix = f"{decoder_prefix}model.0.t_conv.blind."
                decoder_trainable = name.startswith(decoder_prefix) and not name.startswith(
                    blind_prefix
                )
            trainable = name.startswith(skip_prefix) or decoder_trainable
        parameter.requires_grad_(trainable)
        if trainable:
            active.append(name)
    if not active:
        raise RuntimeError(f"No trainable parameters for stage {stage}")
    return active


def _decoder_tail_keys(state: Mapping[str, torch.Tensor], prefix: str, start: int):
    """Select the ordered weights copied by the original resave_ckpt.py."""
    keys = []
    for key in state:
        if not key.startswith(prefix) or ".t_conv.skip." in key:
            continue
        index = int(key[len(prefix) :].split(".", 1)[0])
        if index >= start:
            keys.append(key)
    return keys


def _copy_decoder_tail(
    state: dict[str, torch.Tensor], source: str, target: str, source_start: int, target_start: int
) -> None:
    source_keys = _decoder_tail_keys(state, source, source_start)
    target_keys = _decoder_tail_keys(state, target, target_start)
    if len(source_keys) != len(target_keys):
        raise ValueError(
            f"Decoder handoff has {len(source_keys)} source and "
            f"{len(target_keys)} destination weights"
        )
    for source_key, target_key in zip(source_keys, target_keys):
        if state[source_key].shape != state[target_key].shape:
            raise ValueError(
                f"Decoder handoff shape mismatch: {source_key} "
                f"{tuple(state[source_key].shape)} -> {target_key} "
                f"{tuple(state[target_key].shape)}"
            )
        state[target_key] = state[source_key].clone()


def initialize_from_previous(model: DAC, source: Mapping[str, torch.Tensor]) -> None:
    """Initialize a new stage from a completed previous-stage generator.

    The core stage starts from scratch. Mid starts from core and high starts
    from mid. A stage handoff copies the trained decoder tail into the next
    band's blind decoder, matching the historical resave_ckpt.py behavior.
    """
    if model.stage not in ("mid", "high"):
        raise ValueError("Only mid and high stages can import previous-stage weights")

    target = model.state_dict()
    fresh = ("skip_aes.0.",) if model.stage == "mid" else ("skip_aes.1.", "multidecoders.1.")
    for key, target_value in target.items():
        if key.startswith(fresh):
            continue
        if key not in source:
            raise ValueError(f"Previous-stage checkpoint is missing {key}")
        if source[key].shape != target_value.shape:
            raise ValueError(
                f"Previous-stage weight {key} has shape {tuple(source[key].shape)}, "
                f"expected {tuple(target_value.shape)}"
            )
        target[key] = source[key].clone()

    if model.stage == "mid":
        _copy_decoder_tail(target, "multidecoders.0.model.", "decoder.model.", 0, 1)
    else:
        _copy_decoder_tail(target, "multidecoders.0.model.", "multidecoders.1.model.", 1, 0)
    model.load_state_dict(target, strict=True)


def stage_target(audio: torch.Tensor, stage: str, splitter, scaler: float = 1.0):
    """Return the historical loss target for a training stage."""
    if stage == "core":
        return splitter(audio).sum(0)
    band = 1 if stage == "mid" else 2 if stage == "high" else None
    if band is None:
        raise ValueError(f"Unknown training stage: {stage}")
    return splitter(audio)[band:].sum(0) * scaler
