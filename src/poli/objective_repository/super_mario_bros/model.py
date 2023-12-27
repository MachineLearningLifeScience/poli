"""
A categorical VAE that can train on Mario.
"""
from typing import Tuple
from pathlib import Path

import torch
from torch.distributions import Distribution, Normal, Categorical, kl_divergence
import torch.nn as nn


class VAEMario(nn.Module):
    """
    A VAE that decodes to the Categorical distribution
    on "sentences" of shape (h, w).
    """

    def __init__(
        self,
        w: int = 14,
        h: int = 14,
        z_dim: int = 2,
        n_sprites: int = 11,
        device: str = None,
    ):
        super(VAEMario, self).__init__()
        self.w = w
        self.h = h
        self.n_sprites = n_sprites
        self.input_dim = w * h * n_sprites  # for flattening
        self.z_dim = z_dim
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        ).to(self.device)
        self.enc_mu = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)
        self.enc_var = nn.Sequential(nn.Linear(128, z_dim)).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, self.input_dim),
        ).to(self.device)

        # The VAE prior on latent codes. Only used for the KL term in
        # the ELBO loss.
        self.p_z = Normal(
            torch.zeros(self.z_dim, device=self.device),
            torch.ones(self.z_dim, device=self.device),
        )

        # print(self)

    def encode(self, x: torch.Tensor) -> Normal:
        """
        An encoding function that returns the normal distribution
        q(z|x) for some data x.

        It flattens x after the first dimension, passes it through
        the encoder networks which parametrize the mean and log-variance
        of the Normal, and returns the distribution.
        """
        x = x.view(-1, self.input_dim).to(self.device)
        result = self.encoder(x)
        mu = self.enc_mu(result)
        log_var = self.enc_var(result)

        return Normal(mu, torch.exp(0.5 * log_var))

    def decode(self, z: torch.Tensor) -> Categorical:
        """
        A decoding function that returns the Categorical distribution
        p(x|z) for some latent codes z.

        It passes it through the decoder network, which parametrizes
        the logits of the Categorical distribution of shape (h, w).
        """
        logits = self.decoder(z)
        p_x_given_z = Categorical(
            logits=logits.reshape(-1, self.h, self.w, self.n_sprites)
        )

        return p_x_given_z

    def forward(self, x: torch.Tensor) -> Tuple[Normal, Categorical]:
        """
        A forward pass for some data x, returning the tuple
        [q(z|x), p(x|z)] where the latent codes in the second
        distribution are sampled from the first one.
        """
        q_z_given_x = self.encode(x.to(self.device))

        z = q_z_given_x.rsample()

        p_x_given_z = self.decode(z.to(self.device))

        return [q_z_given_x, p_x_given_z]

    def elbo_loss_function(
        self, x: torch.Tensor, q_z_given_x: Distribution, p_x_given_z: Distribution
    ) -> torch.Tensor:
        """
        The ELBO (Evidence Lower Bound) loss for the VAE,
        which is a linear combination of the reconconstruction
        loss (i.e. the negative log likelihood of the data), plus
        a Kullback-Leibler regularization term which shapes the
        approximate posterior q(z|x) to be close to the prior p(z),
        which we take as the unit Gaussian in latent space.
        """
        x_ = x.to(self.device).argmax(dim=1)  # assuming x is bchw.
        rec_loss = -p_x_given_z.log_prob(x_).sum(dim=(1, 2))  # b
        kld = kl_divergence(q_z_given_x, self.p_z).sum(dim=1)  # b

        return (rec_loss + kld).mean()


def load_example_model(path_to_state_dict: Path) -> VAEMario:
    """
    Loads a pretrained model from the given path.
    """
    vae = VAEMario()
    vae.load_state_dict(
        torch.load(path_to_state_dict, map_location=torch.device("cpu"))
    )
    return vae


if __name__ == "__main__":
    vae = VAEMario()
    print(vae)
