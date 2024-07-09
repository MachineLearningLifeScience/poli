from __future__ import annotations

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox

from poli.core.black_box_information import BlackBoxInformation
from poli.objective_repository.ehrlich._construct_feasibility_matrix import (
    _construct_sparse_transition_matrix,
)
from poli.objective_repository.ehrlich.information import (
    ehrlich_info,
)

from poli.core.util.proteins.defaults import AMINO_ACIDS


class EhrlichBlackBox(AbstractBlackBox):
    def __init__(
        self,
        sequence_length: int,
        motif_length: int,
        n_motifs: int,
        quantization: int = 1,
        alphabet: list[str] = AMINO_ACIDS,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ):
        super().__init__(batch_size, parallelize, num_workers, evaluation_budget)
        self.alphabet = alphabet
        self.sequence_length = sequence_length

        if motif_length * n_motifs > sequence_length:
            raise ValueError(
                "The total length of the motifs is greater than the sequence length."
            )

        if not (1 <= quantization <= motif_length) or motif_length % quantization != 0:
            raise ValueError(
                "The quantization parameter must be between 1 and the motif length, "
                "and the motif length must be divisible by the quantization."
            )

        self.motif_length = motif_length
        self.n_motifs = n_motifs
        self.quantization = quantization

        self.sparse_transition_matrix = _construct_sparse_transition_matrix(
            len(alphabet)
        )

        self.motifs = self.construct_random_motifs(
            motif_length=motif_length, n_motifs=n_motifs
        )
        self.offsets = self.construct_random_offsets(
            motif_length=motif_length, n_motifs=n_motifs
        )

    def _sample_random_sequence(
        self,
        length: int | None = None,
        random_state: np.random.RandomState | None = None,
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

        sequence = self.alphabet[random_state.randint(len(self.alphabet))]
        current_state = self.alphabet.index(sequence)

        for _ in range(length - 1):
            next_state = random_state.choice(
                len(self.alphabet), p=self.sparse_transition_matrix[current_state]
            )
            if not repeating_allowed:
                while next_state == current_state:
                    next_state = random_state.choice(
                        len(self.alphabet),
                        p=self.sparse_transition_matrix[current_state],
                    )
            sequence += self.alphabet[next_state]
            current_state = next_state

        return sequence

    def _is_feasible(self, sequence: str) -> bool:
        """
        Checks whether a sequence (str) is feasible under the sparse transition
        matrix. This is done by looping through the sequence and determining whether
        the transition probabilities are non-zero.
        """
        current_state = self.alphabet.index(sequence[0])
        for i in range(1, len(sequence)):
            next_state = self.alphabet.index(sequence[i])

            if self.sparse_transition_matrix[current_state, next_state] == 0:
                return False
            current_state = next_state

        return True

    def construct_random_motifs(
        self, motif_length: int, n_motifs: int, seed: int = None
    ) -> np.ndarray:
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
                list(sequence[i : (i + 1) * motif_length])
                for i in range(0, len(sequence), motif_length)
            ]
        )

        return motifs

    def construct_random_offsets(
        self,
        motif_length: int,
        n_motifs: int,
        seed: int = None,
    ) -> np.ndarray:
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
        self, motifs: np.ndarray, offsets: np.ndarray
    ) -> np.ndarray:

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

        return np.array(optimal_sequence)

    def _maximal_motif_matches(
        self, sequence: np.ndarray, motif: np.ndarray, offset: np.ndarray
    ) -> int:
        """
        Counts the maximal motif match.
        """
        maximal_match = 0
        for seq_idx in range(len(sequence) - max(offset)):
            matches = 0
            sequence_at_offset = np.array(
                [sequence[seq_idx + offset_value] for offset_value in offset]
            )
            matches = sum(sequence_at_offset == motif)
            print(f"matching {motif} against {sequence_at_offset} gives {matches}")

            maximal_match = max(maximal_match, matches)

        return maximal_match

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
        values = []
        for sequence in x:
            if not self._is_feasible(sequence):
                values.append(-np.inf)
                continue

            value = 1.0
            for motif, offset in zip(self.motifs, self.offsets):
                maximal_matches = self._maximal_motif_matches(sequence, motif, offset)
                value *= (
                    maximal_matches // (self.motif_length / self.quantization)
                ) / self.quantization

            values.append(value)

        return np.array(values).reshape(-1, 1)

    @staticmethod
    def get_black_box_info() -> BlackBoxInformation:
        return ehrlich_info


if __name__ == "__main__":
    for SEED in [1]:
        print(SEED)
        np.random.seed(SEED)
        ehrlich = EhrlichBlackBox(sequence_length=20, motif_length=3, n_motifs=2)
        print(ehrlich.motifs)
        print(ehrlich.offsets)

        optimal_sequence = ehrlich.construct_optimal_solution(
            ehrlich.motifs, ehrlich.offsets
        )
        print(optimal_sequence)

        print(ehrlich(optimal_sequence.reshape(1, -1)))
