"""
Stanton et al.'s [1] implementation of Ehrlich functions using Holo-bench.

References
----------
[1] Stanton, S., Alberstein, R., Frey, N., Watkins, A., & Cho, K. (2024).
    Closed-Form Test Functions for Biophysical Sequence Optimization Algorithms.
    arXiv preprint arXiv:2407.00236. https://arxiv.org/abs/2407.00236
"""

from __future__ import annotations

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.isolation.instancing import get_inner_function
from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli.core.util.seeding import seed_python_numpy_and_torch


class EhrlichHoloBlackBox(AbstractBlackBox):
    """
    Ehrlich functions were proposed by Stanton et al. [1] as a quick-and-easy
    alternative for testing discrete sequence optimizers (with protein
    optimization in mind). They are deviced to

    (i) be easy to query,
    (ii) have feasible and unfeasible sequences,
    (iii) have uninformative random samples (i.e. randomly sampling and evaluating should not be competitive, as many of these should be unfeasible).
    (iv) be maximized when certain motifs are present in the sequence. These motifs can be long-range within the sequence, and are meant to be non-additive.

    Check the references for details on the implementation.

    Parameters
    ----------
    sequence_length : int
        The length of the sequence to be optimized. This length is fixed, and
        _only_ sequences of this length are considered.
    motif_length : int
        The length of the motifs.
    n_motifs : int
        The number of motifs.
    quantization : int, optional
        The quantization parameter. This parameter must be between 1 and the
        motif length, and the motif length must be divisible by the quantization.
        By default, it is None (which corresponds to the motif length).
    noise_std : float, optional
        The noise that gets injected into botorch's SyntheticTestFunction.
        By default, it is 0.0.
    seed : int, optional
        The seed for the random number generator. By default, it is None
        (i.e. a random seed is set using np.random.randint(0, 1000)).
    epistasis_factor : float, optional
        The epistasis factor. By default, it is 0.0.
    return_value_on_unfeasible : float, optional
        The value to be returned when an unfeasible sequence is evaluated.
        By default, it is -np.inf.
    alphabet : list of str, optional
        The alphabet to be used for the sequences. By default, it is the
        of 20 amino acids.
    batch_size : int, optional
        The batch size for the black box. By default, it is None (i.e. all
        sequences are evaluated in a vectorized way).
    parallelize : bool, optional
        Whether to parallelize the evaluation of the black box. By default,
        it is False.
    num_workers : int, optional
        The number of processors used in parallelization.
    evaluation_budget : int, optional
        The evaluation budget for the black box. By default, it is infinite.

    References
    ----------
    [1] Stanton, S., Alberstein, R., Frey, N., Watkins, A., & Cho, K. (2024).
        Closed-Form Test Functions for Biophysical Sequence Optimization Algorithms.
        arXiv preprint arXiv:2407.00236. https://arxiv.org/abs/2407.00236

    """

    def __init__(
        self,
        sequence_length: int,
        motif_length: int,
        n_motifs: int,
        quantization: int | None = None,
        noise_std: float = 0.0,
        seed: int = None,
        epistasis_factor: float = 0.0,
        return_value_on_unfeasible: float = -np.inf,
        alphabet: list[str] = AMINO_ACIDS,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ):
        super().__init__(batch_size, parallelize, num_workers, evaluation_budget)
        self.alphabet = alphabet
        self.sequence_length = sequence_length
        self.return_value_on_unfeasible = return_value_on_unfeasible

        if seed is None:
            # In the case of Ehrlich, it's important we
            # set the seed here, as it will be used in the inner function.
            seed = np.random.randint(1, 1000)

        seed_python_numpy_and_torch(seed)
        self.seed = seed

        if motif_length * n_motifs > sequence_length:
            raise ValueError(
                "The total length of the motifs is greater than the sequence length."
            )

        if quantization is None:
            quantization = motif_length

        if not (1 <= quantization <= motif_length) or motif_length % quantization != 0:
            raise ValueError(
                "The quantization parameter must be between 1 and the motif length, "
                "and the motif length must be divisible by the quantization."
            )

        self.motif_length = motif_length
        self.n_motifs = n_motifs
        self.quantization = quantization

        self.inner_function = get_inner_function(
            isolated_function_name="ehrlich_holo__isolated",
            class_name="EhrlichIsolatedLogic",
            module_to_import="poli.objective_repository.ehrlich_holo.isolated_function",
            force_isolation=force_isolation,
            sequence_length=sequence_length,
            motif_length=motif_length,
            n_motifs=n_motifs,
            quantization=quantization,
            noise_std=noise_std,
            seed=self.seed,
            epistasis_factor=epistasis_factor,
            return_value_on_unfeasible=return_value_on_unfeasible,
            alphabet=alphabet,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

    def initial_solution(self, n_samples: int = 1) -> np.ndarray:
        # This is a sequence of ints.
        initial_solution_as_ints = self.inner_function.initial_solution(n_samples)

        # We convert it to a sequence of strings.
        if n_samples == 1:
            return np.array(
                ["".join([self.alphabet[i] for i in initial_solution_as_ints])]
            )
        else:
            return np.array(
                [
                    "".join([self.alphabet[i] for i in z_i])
                    for z_i in initial_solution_as_ints
                ]
            )

    def random_solution(self) -> np.ndarray:
        random_solution_as_ints = self.inner_function.random_solution

        return np.array(["".join([self.alphabet[i] for i in random_solution_as_ints])])

    def optimal_solution(self) -> np.ndarray:
        optimal_solution_as_ints = self.inner_function.optimal_solution

        return np.array(["".join([self.alphabet[i] for i in optimal_solution_as_ints])])

    def transition_matrix(self) -> np.ndarray:
        return self.inner_function.transition_matrix

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
        """
        Evaluates the sequences in x by checking maximal matches and multiplying.
        """
        return self.inner_function(x, context=context)

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="ehrlich_holo",
            max_sequence_length=self.sequence_length,
            aligned=True,
            fixed_length=True,
            deterministic=True,
            alphabet=self.alphabet,
            log_transform_recommended=False,
            discrete=True,
            padding_token="",
        )


class EhrlichHoloProblemFactory(AbstractProblemFactory):
    """
    A factory for creating Ehrlich functions.

    References
    ----------
    [1] Stanton, S., Alberstein, R., Frey, N., Watkins, A., & Cho, K. (2024).
        Closed-Form Test Functions for Biophysical Sequence Optimization Algorithms.
        arXiv preprint arXiv:2407.00236. https://arxiv.org/abs/2407.00236
    """

    def create(
        self,
        sequence_length: int,
        motif_length: int,
        n_motifs: int,
        quantization: int | None = None,
        noise_std: float = 0.0,
        seed: int = None,
        epistasis_factor: float = 0.0,
        return_value_on_unfeasible: float = -np.inf,
        alphabet: list[str] = AMINO_ACIDS,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        """
        Creates an Ehrlich function problem (containing an Ehrlich black box and
        an initial condition).

        Parameters
        ----------
        sequence_length : int
            The length of the sequence to be optimized. This length is fixed, and
            _only_ sequences of this length are considered.
        motif_length : int
            The length of the motifs.
        n_motifs : int
            The number of motifs.
        quantization : int, optional
            The quantization parameter. This parameter must be between 1 and the
            motif length, and the motif length must be divisible by the quantization.
            By default, it is None (which corresponds to the motif length).
        noise_std : float, optional
            The noise that gets injected into botorch's SyntheticTestFunction.
            By default, it is 0.0.
        seed : int, optional
            The seed for the random number generator. By default, it is None
            (i.e. a random seed is set using np.random.randint(0, 1000)).
        epistasis_factor : float, optional
            The epistasis factor. By default, it is 0.0.
        return_value_on_unfeasible : float, optional
            The value to be returned when an unfeasible sequence is evaluated.
            By default, it is -np.inf.
        alphabet : list of str, optional
            The alphabet to be used for the sequences. By default, it is the
            of 20 amino acids.
        batch_size : int, optional
            The batch size for the black box. By default, it is None (i.e. all
            sequences are evaluated in a vectorized way).
        parallelize : bool, optional
            Whether to parallelize the evaluation of the black box. By default,
            it is False.
        num_workers : int, optional
            The number of processors used in parallelization.
        evaluation_budget : int, optional
            The evaluation budget for the black box. By default, it is infinite.

        References
        ----------
        [1] Stanton, S., Alberstein, R., Frey, N., Watkins, A., & Cho, K. (2024).
            Closed-Form Test Functions for Biophysical Sequence Optimization Algorithms.
            arXiv preprint arXiv:2407.00236. https://arxiv.org/abs/2407.00236
        """
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        f = EhrlichHoloBlackBox(
            sequence_length=sequence_length,
            motif_length=motif_length,
            n_motifs=n_motifs,
            quantization=quantization,
            noise_std=noise_std,
            seed=seed,
            epistasis_factor=epistasis_factor,
            return_value_on_unfeasible=return_value_on_unfeasible,
            alphabet=alphabet,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )
        x0 = f.initial_solution()

        return Problem(f, x0)
