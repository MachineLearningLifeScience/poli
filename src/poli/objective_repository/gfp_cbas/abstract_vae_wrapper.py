"""
Implements a wrapper class for the CBas VAE loading implementation.
Originally implemented in SB's discrete-bo repository.
Only minor modifications conducted by RM.

Creator: SB
Last Changes: RM
"""

import torch


class AbstractVAEWrapper:
    def encode_sequences(self, seqs: [torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("abstract method")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Turns a one hot encoded LxAA vector into a continuous lower dimensional vector.
        :param x:
        :type x:
        :return:
        :rtype:
        """
        raise NotImplementedError("abstract method")

    def decode(self, z: torch.Tensor, grad=False) -> torch.Tensor:
        """
        Returns an NxLxAA tensor.
        :param z:
        :type z:
        :param grad:
        :type grad:
        :return:
        :rtype:
        """
        raise NotImplementedError("abstract method")

    def transform_sequences_to_atoms(self, x: [torch.Tensor]) -> torch.Tensor:
        """
        Turns a sequence into a Dirac distribution around the sequence.
        NO ZERO-HANDLING!
        :param x:
            a list of sequences encoded according to the problem alphabet
        :type x:
        :return:
        Returns an NxLxAA tensor.
        :rtype:
        """
        raise NotImplementedError("abstract method")

    def decode_samples_from_encoder(
        self, x: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        """
        For a single one hot encoded LxAA vector this method samples around the encoded sequence and returns the decoded sequences.
        """
        raise NotImplementedError("abstract method")

    def get_latent_dimensionality(self) -> int:
        raise NotImplementedError("abstract method")

    def get_max_llh_sequence(self, proba: torch.Tensor):
        raise NotImplementedError("abstract method")

    def get_wildtype_distribution(self):
        """Returns a distribution to represent the wild-type."""
        raise NotImplementedError("abstract method")
