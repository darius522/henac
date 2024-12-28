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
from dac.nn.layers import WNConvTranspose1d
from dac.nn.quantize import ResidualVectorQuantize
from dac.model.dac_skip import DACSkip
from plots.plot import residual_plots


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        
        #residual_plots(x, y, 0)
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

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
        skips = []
        for i, m in enumerate(self.block):
            x = m(x)
            if isinstance(m, EncoderBlock):
                skips.append(x.clone())
        skips = list(reversed(skips))
        return x, skips
    

class MergerDecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        
        self.skip = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
            output_padding=(stride % 2) if stride > 1 else 0,
        ))
        self.blind = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
            input_dim,
            output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
            output_padding=(stride % 2) if stride > 1 else 0,
        ))
        
    def forward(self, x_skip, x_blind):
        mn = min(x_skip.shape[-1], x_blind.shape[-1])
        return self.blind(x_blind[..., :mn])# + self.skip(x_skip[..., :mn])
        

class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, merger_block: bool = False):
        super().__init__()
        
        if merger_block:
            self.t_conv = MergerDecoderBlock(input_dim=input_dim, output_dim=output_dim, stride=stride)
        else:
            self.t_conv = WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=(stride % 2) if stride > 1 else 0,
            )

        self.block = nn.Sequential(
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x, x_blind=None):
        if x_blind is not None and isinstance(self.t_conv, MergerDecoderBlock):
            x = self.t_conv(x, x_blind)
        else:
            x = self.t_conv(x)
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        skip: bool = False
    ):
        super().__init__()

        # # Add first conv layer
        # layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]
        layers = []

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            if skip and i == 0:  # if skip+add input
                merger = True
            else:
                merger = False
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride, merger_block=merger)]
                    
        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x, blind_level=None, x_blind=None):
        for i, m in enumerate(self.model):
            if isinstance(m, DecoderBlock) and x_blind is not None:
                x = m(x, x_blind)
            else:
                x = m(x)
            if i == blind_level: # optional: bail early for blind features
                return x
        return x


class DAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        skip_args: dict = dict(),
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )
        from copy import deepcopy
        self.skip_aes = nn.ModuleList([])
        self.skip_aes.append(
            deepcopy(
                DACSkip(
                    encoder_dim=latent_dim//2,
                    latent_dim=latent_dim,
                    codebook_size=codebook_size,
                    quantizer_dropout=quantizer_dropout,
                    **skip_args
                )
            )
        )
        self.multidecoders = nn.ModuleList([])
        self.multidecoders.append(
            Decoder(
                latent_dim//2,
                latent_dim//2,
                decoder_rates[1:],
                skip=True
            )
        )

        self.decoder = Decoder(
            latent_dim,
            latent_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
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
        z_main, skip_feat = self.encoder(audio_data)
        z_main, codes, latents, commitment_loss, codebook_loss, bitrate_loss = self.quantizer(
            z_main, n_quantizers
        )
        skip_out = self.skip_aes[0](skip_feat[1], n_quantizers=n_quantizers)
        # z_skip, codes, latents, commitment_loss, codebook_loss, bitrate_loss = (
        #     skip_out["audio"],
        #     skip_out["codes"],
        #     skip_out["latents"],
        #     skip_out["vq/commitment_loss"],
        #     skip_out["vq/codebook_loss"],
        #     skip_out["vq/bitrate_loss"],
        # )
        return z_main, skip_out["audio"], codes, latents, commitment_loss, codebook_loss, bitrate_loss

    def decode(self, z: torch.Tensor, blind_level: int | None = None):
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
        return self.decoder(z, blind_level)

    def autoencode_skips(self, skips: List, return_output: bool = False, **args):
        outputs = [ae(skip, **args) for skip, ae in zip(skips, self.skip_aes)]
        if return_output:
            return outputs
        return (
            [o["audio"] for o in outputs],
            [o["vq/commitment_loss"] for o in outputs],
            [o["vq/codebook_loss"] for o in outputs],
        )

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
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
        audio_data = self.preprocess(audio_data, sample_rate)
        z_main, z_skip, codes, latents, commitment_loss, codebook_loss, bitrate_loss = self.encode(
            audio_data, n_quantizers
        )

        z_blind = self.decode(z_main, blind_level=0)
        x_skip = self.multidecoders[0](z_skip, blind_level=None, x_blind=z_blind)
        return {
            "audio": x_skip[..., :length],
            "z": z_skip,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
            "vq/bitrate_loss": bitrate_loss,
        }


if __name__ == "__main__":
    import numpy as np
    from functools import partial

    model = DAC().to("cpu")

    for n, m in model.named_modules():
        o = m.extra_repr()
        p = sum([np.prod(p.size()) for p in m.parameters()])
        fn = lambda o, p: o + f" {p/1e6:<.3f}M params."
        setattr(m, "extra_repr", partial(fn, o=o, p=p))
    print(model)
    print("Total # of params: ", sum([np.prod(p.size()) for p in model.parameters()]))

    length = 88200 * 2
    x = torch.randn(1, 1, length).to(model.device)
    x.requires_grad_(True)
    x.retain_grad()

    # Make a forward pass
    out = model(x)["audio"]
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)

    # Create gradient variable
    grad = torch.zeros_like(out)
    grad[:, :, grad.shape[-1] // 2] = 1

    # Make a backward pass
    out.backward(grad)

    # Check non-zero values
    gradmap = x.grad.squeeze(0)
    gradmap = (gradmap != 0).sum(0)  # sum across features
    rf = (gradmap != 0).sum()

    print(f"Receptive field: {rf.item()}")

    x = AudioSignal(torch.randn(1, 1, 44100 * 60), 44100)
    model.decompress(model.compress(x, verbose=True), verbose=True)
