"""
This test module contains all the different instructions
on how to run the different objective functions that are
available poli's objective repository.
"""


def test_white_noise_example():
    import numpy as np
    from poli import objective_factory

    # How to create
    problem = objective_factory.create(name="white_noise")
    f, x0 = problem.black_box, problem.x0
    y0 = f(x0)

    # Example input:
    x = np.array([["1", "2", "3"]])  # must be of shape [b, L], in this case [1, 3].

    # Querying:
    print(f(x))


def test_aloha_example():
    import numpy as np
    from poli import objective_factory

    # How to create
    problem = objective_factory.create(name="aloha")
    f, x0 = problem.black_box, problem.x0

    # Example input:
    x = np.array(
        [["A", "L", "O", "O", "F"]]
    )  # must be of shape [b, L], in this case [1, 3].

    # Querying:
    y = f(x)
    print(y)  # Should be 3 (A, L, and the first O).
    assert np.isclose(y, 3).all()


def test_qed_example():
    import numpy as np
    from poli import objective_factory

    # How to create
    problem = objective_factory.create(
        name="rdkit_qed",
        string_representation="SMILES",  # it is "SMILES" by default.
        force_register=True,
    )
    f, x0 = problem.black_box, problem.x0

    # Example input: a single carbon
    x = np.array(["C"]).reshape(1, -1)

    # Querying:
    y = f(x)
    print(y)  # Should be close to 0.35978
    assert np.isclose(y, 0.35978494).all()


def test_logp_example():
    import numpy as np
    from poli import objective_factory

    # How to create
    problem = objective_factory.create(
        name="rdkit_logp",
        string_representation="SMILES",  # it is "SMILES" by default.
        force_register=True,
    )
    f, x0 = problem.black_box, problem.x0

    # Example input: a single carbon
    x = np.array(["C"]).reshape(1, -1)

    # Querying:
    y = f(x)
    print(y)  # Should be close to 0.6361
    assert np.isclose(y, 0.6361).all()
