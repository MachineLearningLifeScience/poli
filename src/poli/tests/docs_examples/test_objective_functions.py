"""
This test module contains all the different instructions
on how to run the different objective functions that are
available poli's objective repository.
"""

import pytest


def test_white_noise_example():
    import numpy as np
    from poli.objective_repository import WhiteNoiseProblemFactory, WhiteNoiseBlackBox

    # Creating the black box
    f = WhiteNoiseBlackBox()

    # Creating a problem
    problem = WhiteNoiseProblemFactory().create()
    f, x0 = problem.black_box, problem.x0

    # Example input:
    x = np.array([["1", "2", "3"]])  # must be of shape [b, L], in this case [1, 3].

    # Querying:
    print(f(x))


def test_aloha_example():
    import numpy as np
    from poli.objective_repository import AlohaProblemFactory, AlohaBlackBox

    # Creating the black box
    f = AlohaBlackBox()

    # Creating a problem
    problem = AlohaProblemFactory().create()
    f, x0 = problem.black_box, problem.x0

    # Example input:
    x = np.array(
        [["A", "L", "O", "O", "F"]]
    )  # must be of shape [b, L], in this case [1, 5].

    # Querying:
    print(f(x))  # Should be 3 (A, L, and the first O).

    # Querying:
    y = f(x)
    print(y)  # Should be 3 (A, L, and the first O).
    assert np.isclose(y, 3).all()


def test_toy_continuous_example():
    import numpy as np
    from poli.objective_repository import (
        ToyContinuousBlackBox,
        ToyContinuousProblemFactory,
    )

    function_name = "ackley_function_01"
    n_dimensions = 2

    # Creating the black box
    f = ToyContinuousBlackBox(
        function_name=function_name,
        n_dimensions=n_dimensions,
    )

    # Creating a problem
    problem = ToyContinuousProblemFactory().create(
        function_name=function_name,
        n_dimensions=n_dimensions,
    )
    f, x0 = problem.black_box, problem.x0

    # Example input:
    x = np.array([[0.5, 0.5]])  # must be of shape [b, L], in this case [1, 2].

    # Querying:
    print(f(x))

    problem = ToyContinuousProblemFactory().create(
        function_name="camelback_2d",
        embed_in=30,  #  This will create a function that takes 30d input values
    )


def test_qed_example():
    import numpy as np
    from poli.objective_repository import QEDProblemFactory, QEDBlackBox

    # Creating the black box
    f = QEDBlackBox(string_representation="SELFIES")

    # Creating a problem
    problem = QEDProblemFactory().create(string_representation="SELFIES")
    f, x0 = problem.black_box, problem.x0

    # Example input: a single carbon
    x = np.array([["[C]"]])

    # Querying:
    y = f(x)
    print(y)  # Should be close to 0.35978
    assert np.isclose(y, 0.35978494).all()


def test_logp_example():
    import numpy as np
    from poli.objective_repository import LogPProblemFactory, LogPBlackBox

    # Creating the black box
    f = LogPBlackBox(string_representation="SMILES")

    # Creating a problem
    problem = LogPProblemFactory().create(string_representation="SMILES")
    f, x0 = problem.black_box, problem.x0

    # Example input: a single carbon
    x = np.array(["C"]).reshape(1, -1)

    # Querying:
    y = f(x)
    print(y)  # Should be close to 0.6361
    assert np.isclose(y, 0.6361).all()


def test_dockstring_example():
    import numpy as np
    from poli.objective_repository import DockstringProblemFactory, DockstringBlackBox

    # Creating the black box
    f = DockstringBlackBox(target_name="DRD2")

    # Creating a problem
    problem = DockstringProblemFactory().create(target_name="DRD2")
    f, x0 = problem.black_box, problem.x0

    # Example input: risperidone
    x = np.array(["CC1=C(C(=O)N2CCCCC2=N1)CCN3CCC(CC3)C4=NOC5=C4C=CC(=C5)F"])

    # Querying:
    y = f(x)
    print(y)  # Should be 11.9
    assert np.isclose(y, 11.9).all()


def test_drd3_docking_example():
    # TODO: for this one, we need autodock vina and other stronger dependencies
    # that can't be handled by conda. We should skip this test for now.
    pytest.skip()
    import numpy as np
    from poli.objective_repository import DRD3ProblemFactory, DRD3BlackBox

    # Creating the black box
    f = DRD3BlackBox(string_representation="SMILES", force_isolation=True)

    # Creating a problem
    problem = DRD3ProblemFactory().create(
        string_representation="SMILES", force_isolation=True
    )
    f, x0 = problem.black_box, problem.x0

    # Example input:
    x = np.array(["c1ccccc1"])

    # Querying:
    y = f(x)
    print(y)  # Should be close to -4.1
    assert np.isclose(y, -4.1).all()


def test_penalized_logp_lambo():
    import numpy as np
    from poli.objective_repository import (
        PenalizedLogPLamboProblemFactory,
        PenalizedLogPLamboBlackBox,
    )

    # Creating the black box
    f = PenalizedLogPLamboBlackBox()

    # Creating a problem
    problem = PenalizedLogPLamboProblemFactory().create()
    f, x0 = problem.black_box, problem.x0

    # Example input: a single carbon
    x = np.array(["C"]).reshape(1, -1)

    # Querying:
    y = f(x)
    print(y)  # Should be close to 0.6361
    assert np.isclose(y, -6.22381305).all()


if __name__ == "__main__":
    test_penalized_logp_lambo()
