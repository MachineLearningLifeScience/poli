"""
Test suite for benchmarks.

This test suite only verifies that the benchmarks can
be created without errors. It does not check the correctness
of the benchmarks. In the case of PMO and Guacamol, those
tests are already covered by the tests in objective_repository.
"""

import pytest


def test_creating_toy_continuous_functions_benchmark():
    from poli.benchmarks import ToyContinuousFunctionsBenchmark

    benchmark = ToyContinuousFunctionsBenchmark()

    for problem in benchmark:
        f, x0 = problem.black_box, problem.x0
        f(x0)


def test_creating_embedded_toy_continuous_functions_benchmark():
    from poli.benchmarks import EmbeddedBranin2D, EmbeddedHartmann6D

    for benchmark in [EmbeddedBranin2D(), EmbeddedHartmann6D()]:
        for problem in benchmark:
            f, x0 = problem.black_box, problem.x0
            f(x0)


def test_creating_guacamol_benchmark():
    from poli.benchmarks import GuacaMolGoalDirectedBenchmark

    benchmark = GuacaMolGoalDirectedBenchmark(string_representation="SELFIES")

    for i, problem in enumerate(benchmark):
        f, x0 = problem.black_box, problem.x0
        if i > 3:
            break


def test_creating_pmo_benchmark():
    from poli.benchmarks import PMOBenchmark

    benchmark = PMOBenchmark(string_representation="SELFIES")

    for i, problem in enumerate(benchmark):
        f, x0 = problem.black_box, problem.x0

        if i > 3:
            break
