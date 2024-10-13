import numpy as np

from poli.core.black_box_information import BlackBoxInformation
from poli.core.lambda_black_box import LambdaBlackBox
from poli.tests.observers.test_observers import SimpleObserver


def test_lambda_black_box_works_without_custom_info():
    def f_(x: np.ndarray) -> np.ndarray:
        return np.zeros((len(x), 1))

    f = LambdaBlackBox(
        function=f_,
    )

    assert f.info is not None
    assert (f(np.ones((10, 10))) == 0.0).all
    assert f.num_evaluations == 10


def test_lambda_black_box_with_custom_info_works():
    def f_(x: np.ndarray) -> np.ndarray:
        return np.zeros((len(x), 1))

    f = LambdaBlackBox(
        function=f_,
        info=BlackBoxInformation(
            name="zero",
            max_sequence_length=np.inf,
            aligned=False,
            fixed_length=False,
            deterministic=True,
            alphabet=None,
            discrete=True,
        ),
    )

    assert f.info is not None
    assert f.info.name == "zero"
    assert f.info.deterministic
    assert (f(np.ones((10, 10))) == 0.0).all
    assert f.num_evaluations == 10


def test_attaching_observer_to_lambda_black_box_works():
    def f_(x: np.ndarray) -> np.ndarray:
        return np.zeros((len(x), 1))

    f = LambdaBlackBox(
        function=f_,
        info=BlackBoxInformation(
            name="zero",
            max_sequence_length=np.inf,
            aligned=False,
            fixed_length=False,
            deterministic=True,
            alphabet=None,
            discrete=True,
        ),
    )

    observer = SimpleObserver()
    observer.initialize_observer(
        problem_setup_info=f.info, caller_info={"experiment_id": "example"}, seed=None
    )

    f.set_observer(observer)

    f(np.ones((10, 10)))

    assert (np.array(observer.results[0]["x"]) == 1.0).all()
    assert (np.array(observer.results[0]["y"]) == 0.0).all()


if __name__ == "__main__":
    test_attaching_observer_to_lambda_black_box_works()
