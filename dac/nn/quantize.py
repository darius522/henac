from typing import Union

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from dac.nn.layers import WNConv1d

from plots.plot import residual_plots


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int, gumbel_softmax: bool = True, tau_max=1.0, tau_min=0.1, tau_decay=5e-4):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        
        self.gumbel_softmax = gumbel_softmax
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.tau_decay = tau_decay  # Decay rate for annealing
        self.register_buffer("tau", torch.tensor(tau_max))  # Initial temperature

    def forward(self, z, step):
        """Quantize the input tensor using a fixed codebook with Gumbel-Softmax relaxation."""
        if step:
            self.anneal_temperature(step)  # Update tau
        z_e = self.in_proj(z)  # (B x D x T)
        z_q, indices, entropy_loss = self.decode_latents(z_e)

        # Compute losses
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        # Use Gumbel-Softmax trick instead of STE
        if self.gumbel_softmax:
            z_q = z_e + (z_q - z_e)  # No gradient detachment (fully differentiable)
        else:
            z_q = z_e + (z_q - z_e).detach()  # Straight-through estimator (original method)

        z_q = self.out_proj(z_q)
        return z_q, commitment_loss, codebook_loss, indices, z_e, entropy_loss

    def decode_latents(self, latents):
        """Gumbel-Softmax based relaxed quantization instead of argmax selection."""
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # (N x D)

        # L2 normalize encodings and codebook (optional for better quantization)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute distances
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )

        if self.gumbel_softmax and self.training:
            # Gumbel-Softmax: Convert distances into a probability distribution
            logits = -dist  # Flip sign since lower distance = higher probability
            gumbel_noise = torch.rand_like(logits).log().neg().log().neg()  # Sample Gumbel noise
            soft_assignments = F.softmax((logits + gumbel_noise) / self.tau, dim=-1)  # Apply temperature

            # Compute weighted sum of codebook vectors
            z_q = soft_assignments @ codebook  # (B*T x D)
            indices = soft_assignments.argmax(dim=-1)  # Get discrete indices (for logging)
            
            # Compute entropy loss
            entropy_loss = -(soft_assignments * soft_assignments.clamp(min=1e-9).log()).sum(dim=-1).mean()
        else:
            # print('Warning: Either not using Gumbel-Softmax or not in training mode, using hard quantization!')
            # Hard argmax selection (standard VQ)
            indices = (-dist).max(1)[1]  # Get closest codebook entry
            z_q = self.embed_code(indices)

            # Empirical entropy (for hard quantization, only used for monitoring)
            counts = torch.bincount(indices.view(-1), minlength=self.codebook_size).float()
            probs = counts / counts.sum()
            entropy_loss = -(probs * probs.clamp(min=1e-9).log()).sum()

        z_q = rearrange(z_q, "(b t) d -> b d t", b=latents.size(0))
        indices = rearrange(indices, "(b t) -> b t", b=latents.size(0))

        return z_q, indices, entropy_loss

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def anneal_temperature(self, step):
        """Update tau using exponential decay."""
        new_tau = self.tau_min + (self.tau_max - self.tau_min) * math.exp(-self.tau_decay * step)
        self.tau.fill_(new_tau)  # Update tau buffer


class ResidualVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        tau_max: float = 1.0,
        tau_min: float = 1.0, 
        tau_decay: float = 5e-4,
        gumbel_softmax: bool = False,
        diff_entropy: bool = False,
        min_n_codebooks: int = 1,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i], gumbel_softmax, tau_max, tau_min, tau_decay)
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout
        self.diff_entropy = diff_entropy
        self.min_n_codebooks = min_n_codebooks

    def forward(self, z, n_quantizers: int = None, step: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
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
        """
        if self.diff_entropy:
            z = torch.diff(z, dim=-1)
            z = F.pad(z, (0, 1), mode="constant", value=0)
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0
        entropy_loss = 0

        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(self.min_n_codebooks, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i, entropy_loss_i = quantizer(
                residual, step=step
            )

            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()
            entropy_loss += (entropy_loss_i * mask).mean()

            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, codes, latents, commitment_loss, codebook_loss, entropy_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)


if __name__ == "__main__":
    rvq = ResidualVectorQuantize(quantizer_dropout=True)
    x = torch.randn(16, 512, 80)
    y = rvq(x)
    print(y["latents"].shape)