"""
This is a suite of tests for the Ehrlich functions
proposed by Stanton et al. (2024).
"""

import numpy as np
import pytest

from poli.objective_repository.ehrlich._construct_feasibility_matrix import (
    _construct_transition_matrix,
)
from poli.repository import EhrlichBlackBox, EhrlichProblemFactory


@pytest.mark.parametrize("size", [3, 5, 8, 10])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_sparse_matrix_construction_is_ergodic_and_aperiodic(size: int, seed: int):
    sparse_transition_matrix = _construct_transition_matrix(size, seed=seed)

    # Checking with the Perron-Frobenius condition
    assert (
        np.linalg.matrix_power(sparse_transition_matrix, (size - 1) ** 2 + 1) > 0.0
    ).all()


@pytest.mark.parametrize("seed", [1, 4])
@pytest.mark.parametrize("sequence_length", [10, 20, 50, 100])
@pytest.mark.parametrize("motif_length", [2, 3, 4, 5])
@pytest.mark.parametrize("n_motifs", [1, 2, 3, 4])
def test_ehrlich_motifs_and_offsets_are_deterministic(
    seed: int, sequence_length: int, motif_length: int, n_motifs: int
):
    if n_motifs * motif_length > sequence_length:
        pytest.skip(
            "The total length of the motifs is greater than the sequence length."
        )
    ehrlich_1 = EhrlichBlackBox(
        sequence_length=sequence_length,
        motif_length=motif_length,
        n_motifs=n_motifs,
        seed=seed,
    )
    ehrlich_2 = EhrlichBlackBox(
        sequence_length=sequence_length,
        motif_length=motif_length,
        n_motifs=n_motifs,
        seed=seed,
    )

    assert np.all(ehrlich_1.motifs == ehrlich_2.motifs)
    assert np.all(ehrlich_1.offsets == ehrlich_2.offsets)


@pytest.mark.parametrize("sequence_length", [10, 20, 50, 100])
@pytest.mark.parametrize("motif_length", [2, 3, 4, 5])
@pytest.mark.parametrize("n_motifs", [1, 2, 3, 4])
def test_ehrlich_gives_different_motifs_for_different_seeds(
    sequence_length, motif_length, n_motifs
):
    if n_motifs * motif_length > sequence_length:
        pytest.skip(
            "The total length of the motifs is greater than the sequence length."
        )

    ehrlich_1 = EhrlichBlackBox(
        sequence_length=sequence_length,
        motif_length=motif_length,
        n_motifs=n_motifs,
        seed=1,
    )
    ehrlich_2 = EhrlichBlackBox(
        sequence_length=sequence_length,
        motif_length=motif_length,
        n_motifs=n_motifs,
        seed=2,
    )

    assert not np.all(ehrlich_1.motifs == ehrlich_2.motifs)


@pytest.mark.parametrize("sequence_length", [10, 20, 50, 100])
@pytest.mark.parametrize("motif_length", [2, 3, 4, 5])
@pytest.mark.parametrize("n_motifs", [1, 2, 3, 4])
def test_ehrlich_function_produces_optimal_sequences(
    sequence_length: int, motif_length: int, n_motifs: int
):
    if n_motifs * motif_length > sequence_length:
        pytest.skip(
            "The total length of the motifs is greater than the sequence length."
        )

    ehrlich = EhrlichBlackBox(
        sequence_length=sequence_length,
        motif_length=motif_length,
        n_motifs=n_motifs,
        seed=1,
    )

    optimal_sequence = ehrlich.construct_optimal_solution()
    assert ehrlich._is_feasible(optimal_sequence)
    assert ehrlich(optimal_sequence) == 1.0


def test_consistency_of_ehrlich_function_motif_matching():
    ehrlich = EhrlichBlackBox(
        sequence_length=10,
        motif_length=3,
        n_motifs=2,
        quantization=3,
        seed=1,
    )

    one_sequence = np.array(["E"] * 10).reshape(1, 10)
    motif_matches = ehrlich._maximal_motif_matches(
        one_sequence, np.array(["E", "V", "D"]), np.array([0, 1, 3])
    )
    assert motif_matches == 1

    another_sequence = "EVEEEEEEEE"
    another_sequence = np.array(list(another_sequence)).reshape(1, 10)
    assert (
        ehrlich._maximal_motif_matches(
            another_sequence, np.array(["E", "V", "D"]), np.array([0, 1, 3])
        )
        == 2
    )

    yet_another_sequence = "EEEEVEDEEE"
    yet_another_sequence = np.array(list(yet_another_sequence)).reshape(1, 10)
    assert (
        ehrlich._maximal_motif_matches(
            yet_another_sequence, np.array(["E", "V", "D"]), np.array([0, 1, 3])
        )
        == 3
    )


def test_creating_a_problem_with_a_factory():
    problem_factory = EhrlichProblemFactory()

    problem = problem_factory.create(
        sequence_length=10,
        motif_length=3,
        n_motifs=2,
        quantization=3,
        seed=1,
    )

    f, x0 = problem.black_box, problem.x0
    _ = f(x0)


def test_creating_with_create():
    from poli import create

    problem = create(
        name="ehrlich",
        sequence_length=10,
        motif_length=3,
        n_motifs=2,
        quantization=3,
        seed=1,
    )
    f, x0 = problem.black_box, problem.x0
    _ = f(x0)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
@pytest.mark.parametrize("unfeasible_value", [-1.0, 0.0])
def test_unfeasible_value(seed: int, unfeasible_value: float):
    f = EhrlichBlackBox(
        sequence_length=3,
        motif_length=2,
        n_motifs=1,
        seed=seed,
        return_value_on_unfeasible=unfeasible_value,
        alphabet=["A", "B", "C"],
    )

    unfeasible_pairs = np.where(f.transition_matrix == 0)
    for i, j in zip(*unfeasible_pairs):
        assert (
            f(np.array([[f.alphabet[i], f.alphabet[j], f.alphabet[i]]]))
            == unfeasible_value
        )
