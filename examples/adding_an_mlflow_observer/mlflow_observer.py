"""
TODO: add how to run this observer, and the dependencies necessary
to run it.
"""
from pathlib import Path

import mlflow
import numpy as np
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.abstract_observer import AbstractObserver


class MlFlowObserver(AbstractObserver):
    def __init__(
        self, tracking_uri: Path, run_id: str = None, experiment_id: str = None
    ) -> None:
        self.step = 0
        self.tracking_uri = tracking_uri
        self.sequences = []

        # Sets up the MLFlow experiment
        # Is there an experiment running at the moment?
        if mlflow.active_run() is not None:
            # If so, continue to log in it.
            mlflow.set_experiment(mlflow.active_run().info.experiment_name)
        else:
            # If not, create a new one.
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.start_run(run_id=run_id, experiment_id=experiment_id)

        super().__init__()

    def initialize_observer(
        self,
        problem_setup_info: ProblemSetupInformation,
        caller_info: object,
        x0: np.ndarray,
        y0: np.ndarray,
        seed: int,
    ) -> None:
        mlflow.log_params(
            {
                "name": problem_setup_info.name,
                "max_sequence_length": problem_setup_info.max_sequence_length,
                "alphabet": problem_setup_info.alphabet,
            }
        )

        mlflow.log_param("x0", x0)
        mlflow.log_param("y0", y0)
        mlflow.log_param("seed", seed)

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        # TODO: do we need to run this one at a time?
        # TODO: How can we log the sequences?
        mlflow.log_metric("y", y, step=self.step)

        if context is not None:
            for key, value in context.items():
                mlflow.log_metric(key, value, step=self.step)

        self.step += 1

    def finish(self) -> None:
        if isinstance(self.tracking_uri, Path):
            with open(self.tracking_uri / "sequences.npy", "wb") as f:
                np.save(f, self.sequences)

            mlflow.log_artifact(self.tracking_uri / "sequences.npy")

        mlflow.end_run()
