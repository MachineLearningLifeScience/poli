from __future__ import annotations

import numpy as np
import torch
from holo.test_functions.closed_form._ehrlich import Ehrlich

from poli.core.abstract_isolated_function import AbstractIsolatedFunction
from poli.core.registry import register_isolated_function
from poli.core.util.proteins.defaults import AMINO_ACIDS


class EhrlichIsolatedLogic(AbstractIsolatedFunction):
    """ """

    def __init__(
        self,
        sequence_length: int,
        motif_length: int,
        n_motifs: int,
        quantization: int | None = None,
        noise_std: float = 0.0,
        seed: int | None = None,
        epistasis_factor: float = 0.0,
        return_value_on_unfeasible: float = -np.inf,
        alphabet: list[str] = AMINO_ACIDS,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ):
        self.sequence_length = sequence_length
        self.motif_length = motif_length
        self.n_motifs = n_motifs
        self.epistasis_factor = epistasis_factor

        if seed is None:
            raise ValueError("The seed parameter must be set.")

        # if quantization is None:
        #     self.quantization = motif_length

        # if not (1 <= quantization <= motif_length) or motif_length % quantization != 0:
        #     raise ValueError(
        #         "The quantization parameter must be between 1 and the motif length, "
        #         "and the motif length must be divisible by the quantization."
        #     )

        self.noise_std = noise_std
        self.quantization = quantization
        self.seed = seed
        self.return_value_on_unfeasible = return_value_on_unfeasible
        self.alphabet = alphabet
        self.parallelize = parallelize
        self.num_workers = num_workers
        self.evaluation_budget = evaluation_budget

        self.inner_ehrlich = Ehrlich(
            num_states=len(alphabet),
            dim=sequence_length,
            num_motifs=n_motifs,
            motif_length=motif_length,
            quantization=quantization,
            noise_std=noise_std,
            negate=False,  # We aim to maximize the function
            random_seed=seed,
        )

    def __call__(self, x: np.ndarray, context: None) -> np.ndarray:
        # First, we transform the strings into integers using the alphabet
        batch_size = x.shape[0]
        x_ = np.array([[self.alphabet.index(c) for c in s] for s in x.flatten()])

        return (
            self.inner_ehrlich(torch.from_numpy(x_))
            .numpy(force=True)
            .reshape(batch_size, 1)
        )


if __name__ == "__main__":
    register_isolated_function(
        EhrlichIsolatedLogic,
        name="ehrlich_holo__isolated",
        conda_environment_name="poli__ehrlich",
    )
