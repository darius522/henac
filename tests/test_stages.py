"""Checks for stage transitions that previously required separate branches."""

import pytest
import torch

from henac.model import DAC
from henac.training_stage import (
    _decoder_tail_keys,
    configure_trainable,
    initialize_from_previous,
)


SKIP_ARGS = {
    0: {"codebook_dim": 16, "n_codebooks": 1, "encoder_rates": [5], "decoder_rates": [5]},
    1: {"codebook_dim": 16, "n_codebooks": 1, "encoder_rates": [16], "decoder_rates": [16]},
}


def model(stage):
    return DAC(
        stage=stage,
        encoder_dim=8,
        latent_dim=128,
        encoder_rates=[2, 2, 5, 20],
        decoder_rates=[20, 5, 2, 2],
        n_codebooks=2,
        codebook_size=8,
        codebook_dim=4,
        skip_args=SKIP_ARGS,
    )


def test_stage_handoffs_copy_the_trained_decoder_tail():
    core = model("core")
    mid = model("mid")
    initialize_from_previous(mid, core.state_dict())
    core_keys = _decoder_tail_keys(core.state_dict(), "multidecoders.0.model.", 0)
    mid_keys = _decoder_tail_keys(mid.state_dict(), "decoder.model.", 1)
    assert len(core_keys) == len(mid_keys)
    for source, target in zip(core_keys, mid_keys):
        torch.testing.assert_close(core.state_dict()[source], mid.state_dict()[target])

    high = model("high")
    initialize_from_previous(high, mid.state_dict())
    mid_keys = _decoder_tail_keys(mid.state_dict(), "multidecoders.0.model.", 1)
    high_keys = _decoder_tail_keys(high.state_dict(), "multidecoders.1.model.", 0)
    assert len(mid_keys) == len(high_keys)
    for source, target in zip(mid_keys, high_keys):
        torch.testing.assert_close(mid.state_dict()[source], high.state_dict()[target])


@pytest.mark.parametrize("stage", ["core", "mid", "high"])
def test_only_current_stage_receives_gradients(stage):
    generator = model(stage)
    active = set(configure_trainable(generator))
    output = generator(torch.randn(1, 1, 800))
    loss = output["audio"].square().mean() + output["vq/commitment_loss"]
    loss.backward()
    parameters = dict(generator.named_parameters())
    assert any(parameters[name].grad is not None for name in active)
    assert all(parameters[name].grad is None for name in parameters.keys() - active)
    if stage == "core":
        assert all(not name.startswith("skip_aes.") for name in active)
    else:
        previous = "encoder." if stage == "mid" else "skip_aes.0."
        assert all(not name.startswith(previous) for name in active)


def test_handoff_rejects_missing_previous_weights():
    source = model("core").state_dict()
    source.pop("encoder.block.0.bias")
    with pytest.raises(ValueError, match="missing"):
        initialize_from_previous(model("mid"), source)
