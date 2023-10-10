def test_minimal_isolation_example():
    """
    Tests the minimal working example from the readme, verbatum.
    """
    from poli import objective_factory
    import numpy as np

    problem_info, f, x0, y0, run_info = objective_factory.create(
        name="sa_tdc",
        force_register=True,
        string_representation="SELFIES",
    )

    x = np.array([["[C]", "[C]", "[C]"]])
    print(f"f({x}) = {f(x)}")


if __name__ == "__main__":
    test_minimal_isolation_example()
