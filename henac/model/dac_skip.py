import math
from typing import List
from typing import Union

import torch
from torch import nn

from henac.nn.quantize import ResidualVectorQuantize
from henac.nn.layers import EncoderBlock, DecoderBlock
from henac.nn.layers import init_weights


class EncoderSkip(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        strides: list = [1, 1, 1],
    ):
        super().__init__()
        self.block = []
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderSkip(nn.Module):
    def __init__(
        self,
        channels,
        rates,
    ):
        super().__init__()

        layers = []

        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DACSkip(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 256,
        encoder_rates: List[int] = [5],
        latent_dim: int = 256,
        decoder_dim: int = 1024,
        decoder_rates: List[int] = [5],
        n_codebooks: int = 2,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 4096,
        quantizer_dropout: float = 0.0,
        sample_rate: int = 32_000,
        tau_decay: float = 5e-4,
        tau_max: float = 1.0,
        gumbel_softmax: bool = False,
        diff_entropy: bool = False,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        self.latent_dim = latent_dim

        self.hop_length = math.prod(encoder_rates)
        self.encoder = EncoderSkip(encoder_dim, encoder_rates)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
            tau_decay=tau_decay,
            tau_max=tau_max,
            tau_min=0.1,
            gumbel_softmax=gumbel_softmax,
            diff_entropy=diff_entropy,
        )

        self.decoder = DecoderSkip(decoder_dim, decoder_rates)
        self.apply(init_weights)

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
        step: int = None,
    ):
        """Encode one skip feature stream and return its quantization terms."""
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss, entropy_loss = self.quantizer(
            z, n_quantizers, step
        )
        return z, codes, latents, commitment_loss, codebook_loss, entropy_loss

    def decode(self, z: torch.Tensor):
        """Decode the quantized skip feature stream."""
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
        step: int = None,
    ):
        """Reconstruct one skip feature stream."""
        length = audio_data.shape[-1]
        z, codes, latents, commitment_loss, codebook_loss, entropy_loss = self.encode(
            audio_data, n_quantizers, step=step
        )
        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
            "vq/entropy_loss": entropy_loss,
        }
