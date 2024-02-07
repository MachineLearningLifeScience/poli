"""Tests for the basic promises regarding observers

This module implements tests for
- whether we can define an observer,
- whether the observer properly logs the
  objective calls, and
- whether we can register observers and run
  them in isolated processes using `set_observer`.
"""

from pathlib import Path
import json

import numpy as np

from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.util.abstract_observer import AbstractObserver

from poli import objective_factory

THIS_DIR = Path(__file__).parent.resolve()


class SimpleObserver(AbstractObserver):
    def __init__(self, experiment_id: str):
        # Creating a unique id for this experiment in
        # particular:
        experiment_id = experiment_id
        self.experiment_id = experiment_id

        # Creating a local directory for the results
        experiment_path = THIS_DIR / "results" / experiment_id
        experiment_path.mkdir(exist_ok=True, parents=True)

        self.experiment_path = experiment_path
        self.results = []

    def initialize_observer(
        self,
        problem_setup_info: ProblemSetupInformation,
        caller_info: object,
        x0: np.ndarray,
        y0: np.ndarray,
        seed: int,
    ) -> object:
        # Saving the metadata for this experiment
        metadata = problem_setup_info.as_dict()

        # Adding the information the user wanted to provide
        # (Recall that this caller info gets forwarded
        # from the objective_factory.create function)
        metadata["caller_info"] = caller_info

        # Saving the initial evaluations and seed
        metadata["x0"] = x0.tolist()
        metadata["y0"] = y0.tolist()
        metadata["seed"] = seed

        # Saving the metadata
        with open(self.experiment_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        # Appending these results to the results file.
        self.results.append({"x": x.tolist(), "y": y.tolist()})


def test_simple_observer_can_be_defined():
    """Tests whether we can define a simple observer"""
    # Creating a simple observer
    observer = SimpleObserver(experiment_id="example")

    # Checking whether the observer is an instance of
    # AbstractObserver
    assert isinstance(observer, AbstractObserver)

    # Remove the results directory.
    # (indirectly testing that the __init__ ran)
    observer.experiment_path.rmdir()


def test_simple_observer_logs_properly():
    """Tests whether the __call__ of a black box function is properly logged"""
    observer = SimpleObserver(experiment_id="example")

    # Creating a black box function
    f, _, _ = objective_factory.create(name="aloha", observer=observer)

    # Evaluating the black box function
    f(np.array([list("MIGUE")]))

    # Checking whether the results were properly logged
    assert observer.results == [{"x": [["M", "I", "G", "U", "E"]], "y": [[0.0]]}]
    (observer.experiment_path / "metadata.json").unlink()
    observer.experiment_path.rmdir()


def test_observer_registration_and_external_instancing():
    """An integration test for the observer registration and external instancing"""
    from poli.core.registry import set_observer, delete_observer_run_script
    from poli.core.util.external_observer import ExternalObserver

    observer = SimpleObserver(experiment_id="example")
    set_observer(
        observer=observer,
        conda_environment_location="poli__chem",
        observer_name="simple__",
    )

    ext = ExternalObserver(observer_name="simple__", experiment_id="example")

    # Creating a black box function
    f, _, _ = objective_factory.create(name="aloha", observer=ext)

    # Evaluating the black box function
    f(np.array([list("MIGUE")]))

    # Checking whether accessing an unexisting attribute raises an error
    # We do it without pytest to avoid having to install it in
    # the poli__chem environment.
    try:
        ext.unexisting_attribute
        raise AssertionError("Should have raised an error")
    except AttributeError:
        pass

    # Cleaning up (and testing whether we can access attributes
    # of the external observer)
    print(ext.experiment_path)
    (ext.experiment_path / "metadata.json").unlink()
    ext.finish()

    # Cleaning up the observer run script
    delete_observer_run_script(observer_name="simple__")
