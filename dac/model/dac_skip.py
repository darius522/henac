import math
from typing import List
from typing import Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn

from .base import CodecMixin
from dac.nn.layers import Snake1d
from dac.nn.layers import WNConv1d
from dac.nn.quantize import ResidualVectorQuantize
from dac.nn.layers import EncoderBlock, DecoderBlock
from dac.nn.layers import init_weights

from plots.plot import residual_plots

class EncoderSkip(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        strides: list = [1,1,1],
        d_latent: int = 64,
    ):
        super().__init__()

        self.block = []
        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # # Create last convolution
        # self.block += [
        #     Snake1d(d_model),
        #     WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        # ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        for i, m in enumerate(self.block):
            x = m(x)
        return x


class DecoderSkip(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
    ):
        super().__init__()

        # # Add first conv layer
        # layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]
        layers = []

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        for i, m in enumerate(self.model):
            x = m(x)
        return x


class DACSkip(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 256,
        encoder_rates: List[int] = [3],
        latent_dim: int = 256,
        decoder_dim: int = 1024,
        decoder_rates: List[int] = [3],
        n_codebooks: int = 4,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 32,
        quantizer_dropout: float = 0.0,
        sample_rate: int = 24000,
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

        self.hop_length = np.prod(encoder_rates)
        self.encoder = EncoderSkip(encoder_dim, encoder_rates, latent_dim)

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

        self.decoder = DecoderSkip(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
        step: int = None,
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss, entropy_loss = self.quantizer(z, n_quantizers, step)
        return z, codes, latents, commitment_loss, codebook_loss, entropy_loss

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
        step: int = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
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
