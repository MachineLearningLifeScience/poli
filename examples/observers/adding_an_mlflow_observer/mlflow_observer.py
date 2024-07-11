"""
This module implements a simple observer using mlflow.

To run this example, you will need to install mlflow:

    pip install mlflow

To check its results, you will need to start a ui:

    mlflow ui --backend-store-uri ./mlruns
"""

from pathlib import Path

import mlflow
import numpy as np

from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.abstract_observer import AbstractObserver


class MlFlowObserver(AbstractObserver):
    def __init__(self, tracking_uri: Path) -> None:
        self.step = 0
        self.tracking_uri = tracking_uri
        self.sequences = []

        super().__init__()

    def initialize_observer(
        self,
        problem_setup_info: BlackBoxInformation,
        caller_info: object,
        seed: int,
    ) -> None:
        if "run_id" in caller_info:
            run_id = caller_info["run_id"]
        else:
            run_id = None

        if "experiment_id" in caller_info:
            experiment_id = caller_info["experiment_id"]
        else:
            experiment_id = None

        # Sets up the MLFlow experiment
        # Is there an experiment running at the moment?
        if mlflow.active_run() is not None:
            # If so, continue to log in it.
            mlflow.set_experiment(mlflow.active_run().info.experiment_name)
        else:
            # If not, create a new one.
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.start_run(run_id=run_id, experiment_id=experiment_id)

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
