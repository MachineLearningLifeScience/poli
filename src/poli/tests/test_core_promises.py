"""This test suite contains the core promises we make to the user."""

import numpy as np


def test_creating_an_instance_of_a_black_box():
    from poli.objective_repository import WhiteNoiseBlackBox

    f = WhiteNoiseBlackBox()
    x = np.array([["1", "2", "3"]])
    y = f(x)


def test_creating_a_problem():
    from poli import create_problem

    white_noise_problem = create_problem(
        name="white_noise",
        seed=42,
        evaluation_budget=100,
    )

    f, x0 = white_noise_problem.black_box, white_noise_problem.x0
    y0 = f(x0)

    f.terminate()


def test_creating_a_black_box_as_an_isolated_process():
    from poli import instance_function_as_isolated_process

    f = instance_function_as_isolated_process(name="white_noise")


def test_instancing_a_black_box_that_requires_isolation():
    from poli.objective_repository.dockstring.register import DockstringBlackBox

    f = DockstringBlackBox(
        target_name="drd2",
        string_representation="SMILES",
    )

    risperidone_smiles = "CC1=C(C(=O)N2CCCCC2=N1)CCN3CCC(CC3)C4=NOC5=C4C=CC(=C5)F"

    # TODO: replace for proper smiles tokenization.
    x0 = np.array([list(risperidone_smiles)])

    print(f(x0))


if __name__ == "__main__":
    test_instancing_a_black_box_that_requires_isolation()
