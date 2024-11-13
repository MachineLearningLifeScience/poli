"""
This module implements Ehrlich functions as black boxes in poli.

Ehrlich functions were proposed by Stanton et al. [1] as a quick-and-easy
alternative for testing discrete sequence optimizers (with protein
optimization in mind). They are deviced to

(i) be easy to query,
(ii) have feasible and unfeasible sequences,
(iii) have uninformative random samples (i.e. randomly sampling
     and evaluating should not be competitive, as many of these
     should be unfeasible).
(iv) be maximized when certain motifs are present in the sequence.
     These motifs can be long-range within the sequence, and are
     meant to be non-additive.

Check the references for details on the implementation.

References
----------
[1] Stanton, S., Alberstein, R., Frey, N., Watkins, A., & Cho, K. (2024).
    Closed-Form Test Functions for Biophysical Sequence Optimization Algorithms.
    arXiv preprint arXiv:2407.00236. https://arxiv.org/abs/2407.00236
"""

from __future__ import annotations

import warnings

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli.core.util.seeding import seed_python_numpy_and_torch
from poli.objective_repository.ehrlich._construct_feasibility_matrix import (
    _construct_transition_matrix,
)


class EhrlichBlackBox(AbstractBlackBox):
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
    seed : int, optional
        The seed for the random number generator. By default, it is None
        (i.e. no seed is set).
    return_value_on_unfeasible : float, optional
        The value to be returned when an unfeasible sequence is evaluated.
        By default, it is -np.inf.
    feasibility_matrix_temperature : float, optional
        The temperature parameter for the feasibility matrix's softmax. By
        default, it is 0.5.
    feasibility_matrix_band_length : int, optional
        The band length for the non-zero values in the feasibility matrix.
        By default, it is None (i.e. if the alphabet size is v, the band
        length is v - 2 * (v // 5)).
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
        seed: int = None,
        return_value_on_unfeasible: float = -np.inf,
        feasibility_matrix_temperature: float = 0.5,
        feasibility_matrix_band_length: int | None = None,
        alphabet: list[str] = AMINO_ACIDS,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ):
        warnings.warn(
            "This EhrlichBlackBox class is different from the original "
            "implementation provided by Stanton et al. If you are interested in "
            "their implementation (for exact comaprisons), please use "
            "EhrlichHoloBlackBox after installing with pip install poli-core[ehrlich].",
            UserWarning,
        )

        super().__init__(batch_size, parallelize, num_workers, evaluation_budget)
        self.alphabet = alphabet
        self.sequence_length = sequence_length
        self.return_value_on_unfeasible = return_value_on_unfeasible

        if seed is not None:
            seed_python_numpy_and_torch(seed)

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

        self.transition_matrix = _construct_transition_matrix(
            size=len(alphabet),
            seed=seed,
            temperature=feasibility_matrix_temperature,
            band_length=feasibility_matrix_band_length,
        )

        self.motifs = self.construct_random_motifs(
            motif_length=motif_length,
            n_motifs=n_motifs,
            seed=seed,
        )
        self.offsets = self.construct_random_offsets(
            motif_length=motif_length,
            n_motifs=n_motifs,
            seed=seed,
        )

    def _sample_random_sequence(
        self,
        length: int | None = None,
        random_state: int | np.random.RandomState | None = None,
        repeating_allowed: bool = True,
    ) -> str:
        """
        Uses the sparse transition matrix to generate a random sequence
        of a given length.
        """
        if length is None:
            length = self.sequence_length

        if random_state is None:
            random_state = np.random.RandomState()

        if isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            pass
        else:
            raise ValueError(
                "The random_state parameter must be an integer or an instance of "
                "np.random.RandomState."
            )

        sequence = self.alphabet[random_state.randint(len(self.alphabet))]
        current_state = self.alphabet.index(sequence)

        for _ in range(length - 1):
            next_state = random_state.choice(
                len(self.alphabet), p=self.transition_matrix[current_state]
            )
            if not repeating_allowed:
                while next_state == current_state:
                    next_state = random_state.choice(
                        len(self.alphabet),
                        p=self.transition_matrix[current_state],
                    )
            sequence += self.alphabet[next_state]
            current_state = next_state

        return sequence

    def _is_feasible(self, sequence: str | np.ndarray) -> bool:
        """
        Checks whether a sequence (str or array of one sequence) is feasible
        under the transition matrix. This is done by looping through
        the sequence and determining whether the transition probabilities
        are non-zero.
        """
        if isinstance(sequence, np.ndarray):
            assert sequence.ndim == 1 or sequence.shape[0] == 1
            sequence = "".join(sequence.flatten())

        current_state = self.alphabet.index(sequence[0])
        for i in range(1, len(sequence)):
            next_state = self.alphabet.index(sequence[i])

            if self.transition_matrix[current_state, next_state] == 0.0:
                return False
            current_state = next_state

        return True

    def construct_random_motifs(
        self, motif_length: int, n_motifs: int, seed: int = None
    ) -> np.ndarray:
        """
        Creates a given number of random motifs of a certain length.
        """
        assert motif_length * n_motifs <= self.sequence_length

        random_state = np.random.RandomState(seed)

        # Sampling a sequence of length motif_length * n_motifs
        sequence = self._sample_random_sequence(
            length=motif_length * n_motifs,
            random_state=random_state,
            repeating_allowed=False,
        )

        # Chunking it into n_motifs
        motifs = np.array(
            [
                list(sequence[i * motif_length : (i + 1) * motif_length])
                for i in range(0, n_motifs)
            ]
        )

        return motifs

    def construct_random_offsets(
        self,
        motif_length: int,
        n_motifs: int,
        seed: int = None,
    ) -> np.ndarray:
        """
        Creates a given number of random offsets for the motifs.
        """
        all_motifs_length = motif_length * n_motifs

        # For each motif, we sample weights in the simplex
        # from a uniform dirichlet
        random_state = np.random.RandomState(seed)

        offsets = []
        for _ in range(n_motifs):
            weights = random_state.dirichlet(np.ones(motif_length - 1))
            _offset_for_motif = [0]
            for weight in weights:
                _offset_for_motif.append(
                    1
                    + np.floor(
                        weight * (self.sequence_length - all_motifs_length) // n_motifs
                    )
                )

            offset_for_motif = np.cumsum(np.array(_offset_for_motif, dtype=int))

            offsets.append(offset_for_motif)

        return np.array(offsets)

    def construct_optimal_solution(
        self, motifs: np.ndarray | None = None, offsets: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Constructs an optimal solution for a given set of motifs and offsets.

        If None are provided, then the motifs and offsets of the black box
        are used.
        """
        if motifs is None:
            motifs = self.motifs

        if offsets is None:
            offsets = self.offsets

        # de-cumsum the offsets
        offsets = np.diff(offsets, prepend=0)
        optimal_sequence = []

        for motif, offset in zip(motifs, offsets):
            # We write first l - 1 characters according to the offsets,
            # and wait to write the last character at the end
            for character, next_offset_value in zip(motif, offset[1:]):
                # Put the current character in the current position all the way through just before the offset
                optimal_sequence += [character] * next_offset_value

            # Write the last character
            optimal_sequence += [motif[-1]]

        # We pad until the sequence length with the last character
        # of the last motif
        optimal_sequence += [motifs[-1][-1]] * (
            self.sequence_length - len(optimal_sequence)
        )

        return np.array(optimal_sequence).reshape(1, -1)

    def _maximal_motif_matches(
        self, sequence: np.ndarray, motif: np.ndarray, offset: np.ndarray
    ) -> int:
        """
        Counts the maximal motif match.
        """
        assert sequence.ndim == 1 or sequence.shape[0] == 1
        sequence = "".join(sequence.flatten())
        maximal_match = 0
        for seq_idx in range(len(sequence) - max(offset)):
            matches = 0
            sequence_at_offset = np.array(
                [sequence[seq_idx + offset_value] for offset_value in offset]
            )
            matches = sum(sequence_at_offset == motif)

            maximal_match = max(maximal_match, matches)

        return maximal_match

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
        """
        Evaluates the sequences in x by checking maximal matches and multiplying.
        """
        values = []
        for sequence in x:
            if not self._is_feasible(sequence):
                values.append(self.return_value_on_unfeasible)
                continue

            value = 1.0
            for motif, offset in zip(self.motifs, self.offsets):
                maximal_matches = self._maximal_motif_matches(sequence, motif, offset)
                value *= (
                    maximal_matches // (self.motif_length / self.quantization)
                ) / self.quantization

            values.append(value)

        return np.array(values).reshape(-1, 1)

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="ehrlich",
            max_sequence_length=self.sequence_length,
            aligned=True,
            fixed_length=True,
            deterministic=True,
            alphabet=self.alphabet,
            log_transform_recommended=False,
            discrete=True,
            padding_token="",
        )


class EhrlichProblemFactory(AbstractProblemFactory):
    """
    A factory for creating Ehrlich functions and initial conditions.

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
        seed: int = None,
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
        seed : int, optional
            The seed for the random number generator. By default, it is None
            (i.e. no seed is set).
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
        warnings.warn(
            "The EhrlichProblemFactory class is deprecated and will be removed in a future version. "
            " Please use EhrlichHoloProblemFactory after installing with pip install poli-core[ehrlich].",
            DeprecationWarning,
        )
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        f = EhrlichBlackBox(
            sequence_length=sequence_length,
            motif_length=motif_length,
            n_motifs=n_motifs,
            quantization=quantization,
            seed=seed,
            return_value_on_unfeasible=return_value_on_unfeasible,
            alphabet=alphabet,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        x0 = np.array([list(f._sample_random_sequence())])

        return Problem(f, x0)
