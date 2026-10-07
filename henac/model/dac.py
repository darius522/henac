import math
from typing import List

import torch
from audiotools.ml import BaseModel
from torch import nn

from henac.nn.layers import EncoderBlock
from henac.nn.layers import ResidualUnit
from henac.nn.layers import Snake1d
from henac.nn.layers import WNConv1d
from henac.nn.layers import WNConvTranspose1d
from henac.nn.layers import init_weights
from henac.nn.quantize import ResidualVectorQuantize
from henac.model.dac_skip import DACSkip


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
    ):
        super().__init__()
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        skips = []
        for m in self.block:
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
            ),
        )
        self.blind = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=(stride % 2) if stride > 1 else 0,
            ),
        )

    def forward(self, x_skip, x_blind):
        if x_skip is None:
            # Stage 1 trains the blind decoder before a skip path exists.
            return self.blind(x_blind)
        mn = min(x_skip.shape[-1], x_blind.shape[-1])
        return self.blind(x_blind[..., :mn]) + self.skip(x_skip[..., :mn])


class DecoderBlock(nn.Module):
    def __init__(
        self, input_dim: int = 16, output_dim: int = 8, stride: int = 1, merger_block: bool = False
    ):
        super().__init__()

        if merger_block:
            self.t_conv = MergerDecoderBlock(
                input_dim=input_dim, output_dim=output_dim, stride=stride
            )
        else:
            self.t_conv = nn.Sequential(
                Snake1d(input_dim),
                WNConvTranspose1d(
                    input_dim,
                    output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                    output_padding=(stride % 2) if stride > 1 else 0,
                ),
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
    def __init__(self, channels, rates, d_out: int = 1, skip: bool = False):
        super().__init__()
        layers = []
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            merger = skip and i == 0
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride, merger_block=merger)]

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
                x_blind = None
            else:
                x = m(x)
            if i == blind_level:
                return x
        return x


class DAC(BaseModel):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 2, 5, 20],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [20, 5, 2, 2],
        n_codebooks: int = 32,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 32_000,
        skip_args: dict = None,
        min_n_codebooks: int = 1,
        stage: str = "core",
    ):
        super().__init__()

        if stage not in ("core", "mid", "high"):
            raise ValueError(f"Unknown training stage: {stage}")
        self.stage = stage

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        if latent_dim != encoder_dim * (2 ** len(encoder_rates)):
            raise ValueError("latent_dim must match the encoder's output channels")
        self.latent_dim = latent_dim

        self.hop_length = math.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
            gumbel_softmax=False,
            min_n_codebooks=min_n_codebooks,
        )
        self.skip_aes = nn.ModuleList([])
        self.multidecoders = nn.ModuleList([])
        skip_args = skip_args or {}
        n_skips = 2 if stage == "high" else 1 if stage == "mid" else 0
        for i in range(max(1, n_skips)):
            enc_dim = latent_dim // (2 ** (i + 1))
            if i < n_skips:
                if i not in skip_args:
                    raise ValueError(f"DAC.skip_args must define path {i} for stage {stage}")
                self.skip_aes.append(
                    DACSkip(
                        encoder_dim=enc_dim,
                        latent_dim=enc_dim * 2,
                        decoder_dim=enc_dim * 2,
                        codebook_size=codebook_size,
                        **skip_args[i],
                    )
                )
            self.multidecoders.append(Decoder(enc_dim, decoder_rates[(i + 1) :], skip=True))

        self.decoder = Decoder(latent_dim, decoder_rates)
        self.apply(init_weights)

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        if sample_rate != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz audio, got {sample_rate} Hz")
        if audio_data.ndim != 3 or audio_data.shape[1] != 1:
            raise ValueError("Expected mono audio with shape [batch, 1, samples]")

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: list | None = None,
        step: int = None,
    ):
        """Encode the paths used by this training stage."""
        n_quantizers = [None, None, None] if n_quantizers is None else n_quantizers
        if len(n_quantizers) != 3:
            raise ValueError("n_quantizers must contain core, mid, and high counts")
        z_main, skip_feat = self.encoder(audio_data)
        z_main, core_codes, _, commitment, codebook, entropy = self.quantizer(
            z_main, n_quantizers=n_quantizers[0], step=step
        )
        feats = {"core": z_main}
        codes = {"core": core_codes}
        if self.stage in ("mid", "high"):
            skip_out_mb = self.skip_aes[0](skip_feat[1], n_quantizers=n_quantizers[1], step=step)
            feats["mb"] = skip_out_mb["audio"]
            codes["mb"] = skip_out_mb["codes"]
            commitment = skip_out_mb["vq/commitment_loss"]
            codebook = skip_out_mb["vq/codebook_loss"]
            entropy = skip_out_mb["vq/entropy_loss"]
        if self.stage == "high":
            skip_out_hb = self.skip_aes[1](skip_feat[2], n_quantizers=n_quantizers[2], step=step)
            feats["hb"] = skip_out_hb["audio"]
            codes["hb"] = skip_out_hb["codes"]
            commitment = skip_out_hb["vq/commitment_loss"]
            codebook = skip_out_hb["vq/codebook_loss"]
            entropy = skip_out_hb["vq/entropy_loss"]
        losses = {
            "commitment_loss": commitment,
            "codebook_loss": codebook,
            "entropy_loss": entropy,
        }
        return feats, codes, losses

    def decode(self, z: torch.Tensor, blind_level: int | None = None):
        """Decode core features, optionally stopping after the first block."""
        return self.decoder(z, blind_level)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: list | None = None,
        step: int = None,
    ):
        """Return the waveform and VQ losses for the selected training stage."""
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        feats, codes, losses = self.encode(audio_data, n_quantizers, step=step)

        xcore_blind = self.decode(feats["core"], blind_level=0)
        if self.stage == "core":
            audio = self.multidecoders[0](None, x_blind=xcore_blind)
        elif self.stage == "mid":
            audio = self.multidecoders[0](feats["mb"], x_blind=xcore_blind)
        else:
            xmb_blind = self.multidecoders[0](feats["mb"], blind_level=0, x_blind=xcore_blind)
            audio = self.multidecoders[1](feats["hb"], x_blind=xmb_blind)
        return {
            "audio": audio[..., :length],
            "codes": codes,
            "feats": feats,
            "vq/commitment_loss": losses["commitment_loss"],
            "vq/codebook_loss": losses["codebook_loss"],
            "vq/entropy_loss": losses["entropy_loss"],
        }

    def infer_bands(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: list | None = None,
        step: int = None,
    ):
        """Decode all paths present in this model before frequency filtering."""
        audio_data = self.preprocess(audio_data, sample_rate)
        feats, codes, _ = self.encode(audio_data, n_quantizers, step=step)
        core_blind = self.decode(feats["core"], blind_level=0)
        if self.stage == "core":
            audio = {"core": self.multidecoders[0](None, x_blind=core_blind)}
        else:
            # Keep the legacy decoder order for reproducible full-path audio.
            core = self.decode(feats["core"])
            mid = self.multidecoders[0](feats["mb"], x_blind=core_blind)
            audio = {"core": core, "mb": mid}
            if self.stage == "high":
                mid_blind = self.multidecoders[0](feats["mb"], blind_level=0, x_blind=core_blind)
                audio["hb"] = self.multidecoders[1](feats["hb"], x_blind=mid_blind)
        return {
            "audio": audio,
            "codes": codes,
            "feats": feats,
        }

    def infer_active_bands(
        self,
        audio_data: torch.Tensor,
        n_quantizers: tuple[int, int, int],
        sample_rate: int | None = None,
    ):
        """Run only the contiguous paths requested for inference."""
        core_count, mid_count, high_count = n_quantizers
        maxima = [self.n_codebooks]
        maxima += [skip.n_codebooks for skip in self.skip_aes]
        maxima += [0] * (3 - len(maxima))
        if not 1 <= core_count <= maxima[0]:
            raise ValueError(f"Core codebooks must be in [1, {maxima[0]}]")
        if not 0 <= mid_count <= maxima[1]:
            raise ValueError(f"Mid codebooks must be in [0, {maxima[1]}]")
        if not 0 <= high_count <= maxima[2] or (high_count and not mid_count):
            raise ValueError("High codebooks require an active mid path")

        audio_data = self.preprocess(audio_data, sample_rate)
        core, skip_features = self.encoder(audio_data)
        core, core_codes, _, _, _, _ = self.quantizer(core, n_quantizers=core_count)
        if self.stage == "core":
            core_blind = self.decode(core, blind_level=0)
            return {
                "audio": {"core": self.multidecoders[0](None, x_blind=core_blind)},
                "codes": {"core": core_codes},
            }

        audio = {"core": self.decode(core)}
        codes = {"core": core_codes}
        if mid_count:
            mid_out = self.skip_aes[0](skip_features[1], n_quantizers=mid_count)
            core_blind = self.decode(core, blind_level=0)
            mid = mid_out["audio"]
            codes["mb"] = mid_out["codes"]
            audio["mb"] = self.multidecoders[0](mid, x_blind=core_blind)
            if high_count:
                high_out = self.skip_aes[1](skip_features[2], n_quantizers=high_count)
                mid_blind = self.multidecoders[0](mid, blind_level=0, x_blind=core_blind)
                codes["hb"] = high_out["codes"]
                audio["hb"] = self.multidecoders[1](high_out["audio"], x_blind=mid_blind)
        return {"audio": audio, "codes": codes}
