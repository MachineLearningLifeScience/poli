def test_minimal_working_example():
    """
    Tests the minimal working example from the readme, verbatum.
    """
    import numpy as np
    from poli import objective_factory

    problem_info, f, x0, y0, run_info = objective_factory.create(name="white_noise")

    x = np.array([["1", "2", "3"]])  # must be of shape [b, L], in this case [1, 3].
    for _ in range(5):
        print(f"f(x) = {f(x)}")
