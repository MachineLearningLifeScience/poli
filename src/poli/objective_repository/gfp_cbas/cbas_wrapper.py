"""
Implements a wrapper class for the CBas VAE loading implementation.
Only minor modifications conducted by second listed authors, all credit to this herculean task go to SB

Creator: SB
Last Changes: RM
"""

__author__ = "Simon Bartels, Richard Michael"


from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm as StandardNormal
from scipy.stats.qmc import Sobol
from torch.nn import functional as F

from poli.objective_repository.gfp_cbas.abstract_vae_wrapper import AbstractVAEWrapper
from poli.objective_repository.gfp_cbas.make_vae import build_vae


class CBASVAEWrapper(AbstractVAEWrapper):
    def __init__(self, AA: int, L: int):
        self.L = L
        self.AA = AA
        model_path = Path(__file__).parent.resolve() / "assets" / "models" / "vae"
        vae_0 = build_vae(
            latent_dim=20,
            n_tokens=self.AA,  # 20,  # TODO: test if this is self.AA?
            seq_length=self.L,
            enc1_units=50,
        )
        # unfortunately the call below is not usable
        # vae_0.load_all_weights()
        print(f"Model Path: {model_path}")
        vae_suffix = "_5k_1"
        vae_0.encoder_.load_weights(
            str(model_path / f"vae_0_encoder_weights{vae_suffix}.h5")
        )
        vae_0.decoder_.load_weights(
            str(model_path / f"vae_0_decoder_weights{vae_suffix}.h5")
        )
        vae_0.vae_.load_weights(str(model_path / f"vae_0_vae_weights{vae_suffix}.h5"))

        self.vae = vae_0
        enc1_units = 50  # according to the CBAS paper
        self._decoder = ConvertedTorchVaeDecoder(
            vae_0, self.AA, self.L, vae_0.latentDim_, enc1_units
        )
        self._encoder = ConvertedTorchVaeEncoder(
            vae_0, self.AA, self.L, vae_0.latentDim_, enc1_units
        )

    def encode(self, x, grad=False):
        # seems like also for this VAE, the dimension of the amino acids is 1
        x_ = torch.flatten(
            torch.permute(torch.reshape(x, [x.shape[0], self.L, self.AA]), [0, 2, 1]),
            start_dim=1,
        )
        if not grad:
            with torch.no_grad():
                # x = self.vae.decoder_.predict(z.numpy())
                return self._encoder(x_)[0]
        else:
            return self._encoder(x_)[0]

    def decode(self, z, grad=False):
        if not grad:
            with torch.no_grad():
                # x = self.vae.decoder_.predict(z.numpy())
                return self._decoder(z)
        else:
            return self._decoder(z)

    def transform_sequences_to_atoms(self, x: torch.Tensor) -> torch.Tensor:
        """
        Turns a sequence into an approximately Dirac distribution around the sequence.
        :param x:
        :type x:
        :return:
        :rtype:
        """
        raise NotImplementedError("decide on how to handle zeros!")

    def get_max_llh_sequence(self, proba):
        assert len(proba.shape) == 3
        mle_x = F.one_hot(proba.argmax(dim=-1), self.AA)
        return mle_x.flatten(start_dim=1)

    def get_latent_dimensionality(self):
        return self.vae.latentDim_

    def decode_samples_from_encoder(
        self, x: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        s = Sobol(d=self.get_latent_dimensionality())
        samples = s.random(num_samples)
        samples = StandardNormal.ppf(samples.flatten()).reshape(samples.shape)
        with torch.no_grad():
            x_ = torch.flatten(x, start_dim=1)
            mu, logvar = self._encoder(x_)
            samples = torch.tensor(samples * np.exp(logvar.numpy() / 2) + mu.numpy())
            return (self.decode(samples, grad=False)).flatten(start_dim=1)


class ConvertedTorchVaeDecoder(torch.nn.Module):
    """
    Module that mimics the keras decoder as specified in the CBAS paper.
    """

    def __init__(self, vae, AA, L, latent_dimensionality, enc1_units):
        super().__init__()
        self.first_part = torch.nn.Sequential(
            OrderedDict(
                [
                    ("d1", torch.nn.Linear(latent_dimensionality, enc1_units)),
                    ("d1act", torch.nn.ELU()),
                    ("d3", torch.nn.Linear(enc1_units, AA * L)),
                ]
            )
        )
        named_children = dict(self.first_part.named_children())
        kt_layers = {layer.name: layer for layer in vae.decoder_.layers}
        for n, u in named_children.items():
            if n not in kt_layers.keys():
                continue
            u.weight.data = torch.as_tensor(
                kt_layers[n].get_weights()[0], dtype=torch.float64
            ).T
            u.bias.data = torch.as_tensor(
                kt_layers[n].get_weights()[1], dtype=torch.float64
            )
        self.d5 = torch.nn.Linear(AA * L, AA)
        self.d5.weight.data = torch.as_tensor(
            kt_layers["d5"].get_weights()[0], dtype=torch.float64
        ).T
        self.d5.bias.data = torch.as_tensor(
            kt_layers["d5"].get_weights()[1], dtype=torch.float64
        )

        self.AA = AA
        self.L = L

    def forward(self, z):
        return torch.nn.Softmax(dim=-1)(
            self.d5(self.first_part(z).reshape([z.shape[0], self.L, self.AA]))
        )


class ConvertedTorchVaeEncoder(torch.nn.Module):
    def __init__(self, vae, AA, L, latent_dimensionality, enc1_units):
        super().__init__()
        self.first_part = nn.Sequential(
            OrderedDict(
                [
                    # fully connected
                    ("e2", torch.nn.Linear(AA * L, enc1_units)),
                    ("e2act", torch.nn.ELU()),
                ]
            )
        )
        self.mu = torch.nn.Linear(enc1_units, latent_dimensionality)
        self.log_var = torch.nn.Linear(enc1_units, latent_dimensionality)
        named_children = dict(self.first_part.named_children())
        w = vae.vae_
        kt_layers = {layer.name: layer for layer in w.layers}
        named_children["e2"].weight.data = torch.as_tensor(
            kt_layers["e2"].get_weights()[0], dtype=torch.float64
        ).T
        named_children["e2"].bias.data = torch.as_tensor(
            kt_layers["e2"].get_weights()[1], dtype=torch.float64
        )
        self.mu.weight.data = torch.as_tensor(
            kt_layers["mu_z"].get_weights()[0], dtype=torch.float64
        ).T
        self.mu.bias.data = torch.as_tensor(
            kt_layers["mu_z"].get_weights()[1], dtype=torch.float64
        )
        self.log_var.weight.data = torch.as_tensor(
            kt_layers["log_var_z"].get_weights()[0], dtype=torch.float64
        ).T
        self.log_var.bias.data = torch.as_tensor(
            kt_layers["log_var_z"].get_weights()[1], dtype=torch.float64
        )

    def forward(self, x):
        t = self.first_part(x)
        return self.mu(t), self.log_var(t)
