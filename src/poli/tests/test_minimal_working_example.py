def test_minimal_working_example_of_problem_creation():
    """
    Tests the minimal working example from the readme, verbatum.
    """
    import numpy as np
    from poli import objective_factory

    white_noise_problem = objective_factory.create_problem(name="white_noise")
    f = white_noise_problem.black_box

    x = np.array([["1", "2", "3"]])  # must be of shape [b, L], in this case [1, 3].
    for _ in range(5):
        print(f"f(x) = {f(x)}")


def test_minimal_working_example_of_black_box_instancing():
    import numpy as np
    from poli.objective_repository import WhiteNoiseBlackBox

    f = WhiteNoiseBlackBox()
    x = np.array([["1", "2", "3"]])
    print(f"f(x) = {f(x)}")
