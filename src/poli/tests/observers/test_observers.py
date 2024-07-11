"""Tests for the basic promises regarding observers

This module implements tests for
- whether we can define an observer,
- whether the observer properly logs the
  objective calls, and
- whether we can register observers and run
  them in isolated processes using `set_observer`.
"""

import json
from pathlib import Path

import numpy as np

from poli import objective_factory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.registry import register_observer
from poli.core.util.abstract_observer import AbstractObserver

THIS_DIR = Path(__file__).parent.resolve()


class SimpleObserver(AbstractObserver):
    def initialize_observer(
        self,
        problem_setup_info: BlackBoxInformation,
        caller_info: object,
        seed: int,
    ) -> object:
        experiment_id = caller_info["experiment_id"]

        self.experiment_id = experiment_id

        # Creating a local directory for the results
        experiment_path = THIS_DIR / "results" / experiment_id
        experiment_path.mkdir(exist_ok=True, parents=True)

        self.experiment_path = experiment_path
        self.results = []
        # Saving the metadata for this experiment
        metadata = problem_setup_info.as_dict()

        # Adding the information the user wanted to provide
        # (Recall that this caller info gets forwarded
        # from the objective_factory.create function)
        metadata["caller_info"] = caller_info
        metadata["seed"] = seed

        # Saving the metadata
        with open(self.experiment_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return None

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        # Appending these results to the results file.
        self.results.append({"x": x.tolist(), "y": y.tolist()})


def test_registering_observer_with_bad_names():
    # TODO[SIMON]: implement
    pass


def test_simple_observer_can_be_defined():
    """Tests whether we can define a simple observer"""
    # Creating a simple observer
    observer = SimpleObserver()

    # Checking whether the observer is an instance of
    # AbstractObserver
    assert isinstance(observer, AbstractObserver)


def test_simple_observer_logs_properly():
    """Tests whether the __call__ of a black box function is properly logged"""
    from poli.core.registry import register_observer

    observer = SimpleObserver()

    register_observer(
        observer=observer,
        conda_environment_location="poli__chem",
        observer_name="simple__",
        set_as_default_observer=False,
    )

    # Creating a black box function
    problem = objective_factory.create(
        name="aloha",
        observer_name="simple__",
        observer_init_info={"experiment_id": "example"},
    )
    f = problem.black_box

    # Evaluating the black box function
    f(np.array([list("MIGUE")]))

    # Checking whether the results were properly logged
    assert f.observer.results == [{"x": [["M", "I", "G", "U", "E"]], "y": [[0.0]]}]
    (f.observer.experiment_path / "metadata.json").unlink()
    f.observer.experiment_path.rmdir()


def test_observer_registration_and_external_instancing():
    """An integration test for the observer registration and external instancing"""
    from poli.core.registry import register_observer

    observer = SimpleObserver()
    register_observer(
        observer=observer.__class__,
        conda_environment_location="poli__chem",
        observer_name="simple__",
        set_as_default_observer=False,
    )

    # Creating a black box function
    problem = objective_factory.create(
        name="aloha",
        observer_name="simple__",
        observer_init_info={"experiment_id": "example"},
    )
    f = problem.black_box

    # Evaluating the black box function
    f(np.array([list("MIGUE")]))

    # Checking whether accessing an unexisting attribute raises an error
    # We do it without pytest to avoid having to install it in
    # the poli__chem environment.
    f.observer  # The same as problem.observer._observer
    try:
        f.observer.unexisting_attribute
    except AttributeError:
        pass

    # Cleaning up (and testing whether we can access attributes
    # of the external observer)
    print(f.observer.experiment_path)
    (f.observer.experiment_path / "metadata.json").unlink()
    f.observer.finish()


def test_multiple_observer_registration():
    observer = SimpleObserver()
    register_observer(
        observer=observer,
        conda_environment_location="poli__chem",
        observer_name="simple__",
        set_as_default_observer=False,
    )

    observer_2 = SimpleObserver()
    register_observer(
        observer=observer_2,
        conda_environment_location="poli__chem",
        observer_name="simple_2__",
        set_as_default_observer=False,
    )

    # Creating a black box function
    problem_1 = objective_factory.create(
        name="aloha",
        observer_name="simple__",
        observer_init_info={"experiment_id": "example"},
    )
    problem_2 = objective_factory.create(
        name="aloha",
        observer_name="simple_2__",
        observer_init_info={"experiment_id": "example_2"},
    )

    # Evaluating the black box function
    problem_1.black_box(np.array([list("MIGUE")]))
    problem_2.black_box(np.array([list("MIGUE")]))

    # Cleaning up (and testing whether we can access attributes
    # of the external observer)
    (problem_1.observer._observer.experiment_path / "metadata.json").unlink()
    (problem_2.observer._observer.experiment_path / "metadata.json").unlink()
    problem_1.observer._observer.finish()
    problem_2.observer._observer.finish()


def test_attaching_an_observer_to_a_black_box():
    from poli.repository import ToyContinuousBlackBox

    f = ToyContinuousBlackBox(
        function_name="ackley_function_01",
        n_dimensions=10,
    )

    observer = SimpleObserver()

    f.set_observer(observer)

    observer.initialize_observer(f.info, {"experiment_id": "attaching"}, seed=0)

    f(np.array([0.0] * 10).reshape(1, 10))

    assert len(observer.results[0]["x"]) == 1
