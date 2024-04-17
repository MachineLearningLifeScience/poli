"""
Test suite for benchmarks
"""


def test_creating_toy_continuous_functions_benchmark():
    from poli.benchmarks import ToyContinuousFunctionsBenchmark

    benchmark = ToyContinuousFunctionsBenchmark()

    for problem in benchmark:
        f, x0 = problem.black_box, problem.x0
        f(x0)


def test_creating_guacamol_benchmark():
    from poli.benchmarks import GuacamolGoalOrientedBenchmark

    benchmark = GuacamolGoalOrientedBenchmark(string_representation="SELFIES")

    for problem in benchmark:
        f, x0 = problem.black_box, problem.x0
        f(x0)


def test_creating_pmo_benchmark():
    from poli.benchmarks import PMOBenchmark

    benchmark = PMOBenchmark(string_representation="SELFIES")

    for problem in benchmark:
        f, x0 = problem.black_box, problem.x0
        f(x0)


if __name__ == "__main__":
    test_creating_pmo_benchmark()
