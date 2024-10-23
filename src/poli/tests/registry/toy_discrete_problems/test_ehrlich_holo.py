import pytest


@pytest.mark.poli__ehrlich_holo
def test_ehrlich_holo_factory():
    from poli.objective_repository import EhrlichHoloProblemFactory

    problem = EhrlichHoloProblemFactory().create(
        sequence_length=10,
        motif_length=3,
        n_motifs=2,
        epistasis_factor=0.5,
    )
    f, x0 = problem.black_box, problem.x0
    print(f(x0))


@pytest.mark.poli__ehrlich_holo
def test_ehrlich_holo_builds_and_queries():
    from poli.objective_repository import EhrlichHoloBlackBox

    black_box = EhrlichHoloBlackBox(
        sequence_length=10,
        motif_length=3,
        n_motifs=2,
        epistasis_factor=0.5,
    )
    x0 = black_box.initial_solution()
    print(black_box(x0))

    x_final = black_box.optimal_solution()
    print(black_box(x_final))


"""
This test is currently not working on CI. The reason
seems to be something related to the networking of
the CI environment, and the fact that we are querying
properties/methods of the underlying ExternalIsolatedFunction.

These issues will be addressed by the new isolation interface,
and if the user installs this black box with
`pip install poli[ehrlich]` they won't face these issues.

The error we currently get is the following:
multiprocessing.context.AuthenticationError: digest sent was rejected

Testing it locally seems to work well.
TODO: install in Colab and test.
"""
# @pytest.mark.poli__ehrlich_holo
# def test_ehrlich_holo_works_on_isolation():
#     from poli.objective_repository import EhrlichHoloBlackBox

#     black_box = EhrlichHoloBlackBox(
#         sequence_length=10,
#         motif_length=3,
#         n_motifs=2,
#         epistasis_factor=0.0,
#         force_isolation=True,
#     )
#     x0 = black_box.initial_solution(100)
#     print(black_box(x0))

#     x_final = black_box.optimal_solution()
#     print(black_box(x_final))


@pytest.mark.poli__ehrlich_holo
def test_ehrlich_seed_determinism():
    from poli.objective_repository import EhrlichHoloBlackBox

    black_box = EhrlichHoloBlackBox(
        sequence_length=10,
        motif_length=3,
        n_motifs=2,
        epistasis_factor=0.0,
        seed=42,
    )
    x0 = black_box.initial_solution()
    print(black_box(x0))

    black_box_2 = EhrlichHoloBlackBox(
        sequence_length=10,
        motif_length=3,
        n_motifs=2,
        epistasis_factor=0.0,
        seed=42,
    )
    x0_2 = black_box.initial_solution()
    print(black_box_2(x0_2))

    assert (black_box(x0) == black_box_2(x0_2)).all()
    assert (x0 == x0_2).all()
