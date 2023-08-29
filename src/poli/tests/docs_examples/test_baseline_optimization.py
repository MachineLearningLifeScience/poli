import pytest


def test_optimizing_aloha():
    """
    If poli_baselines is available, this test checks
    whether we can optimize the aloha problem.
    """
    from poli import objective_factory
    from poli.core.registry import get_problems

    _ = pytest.importorskip("poli_baselines")

    from poli_baselines.solvers.simple.random_mutation import RandomMutation

    assert "aloha" in get_problems()

    # Creating an instance of the problem
    problem_info, f, x0, y0, run_info = objective_factory.create(
        name="aloha", caller_info=None, observer=None
    )

    # Creating an instance of the solver
    solver = RandomMutation(
        black_box=f,
        x0=x0,
        y0=y0,
        alphabet=problem_info.get_alphabet(),
    )

    # Running the optimization for 1000 steps,
    # breaking if we find a performance above 5.0,
    # and printing a small summary at each step.
    solver.solve(max_iter=1000, break_at_performance=5.0)
    solver.get_best_solution()
