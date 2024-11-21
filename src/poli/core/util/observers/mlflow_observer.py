from pathlib import Path

import mlflow
import numpy as np

from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.abstract_observer import AbstractObserver

TRACKING_URI = "tracking_uri"
OBJECTIVE = "OBJECTIVE"
SEQUENCE = "SEQUENCE"
SEED = "SEED"


class MLFlowObserver(AbstractObserver):
    """
    This observer uses mlflow as a backend.
    """

    def __init__(self, tracking_uri: Path = None):
        self.step = 0
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        for n in range(y.shape[0]):
            self.step += 1
            mlflow.log_metrics(
                {OBJECTIVE + str(i): y[n, i].item() for i in range(y.shape[1])},
                step=self.step,
            )
            # with mlflow it's unfortunately not so easy to log sequences
            mlflow.log_param(str(self.step) + SEQUENCE, x[n, ...])

    def log(self, algorithm_info: dict):
        mlflow.log_metrics(algorithm_info, step=self.step)

    def initialize_observer(
        self,
        problem_setup_info: BlackBoxInformation,
        caller_info: dict,
        seed: int,
    ) -> object:
        tracking_uri = caller_info.pop(TRACKING_URI, None)
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        experiment = mlflow.set_experiment(
            experiment_name=problem_setup_info.get_problem_name()
        )
        run = mlflow.start_run(experiment_id=experiment.experiment_id)
        mlflow.set_tag(SEED, str(seed))
        mlflow.set_tags(caller_info)
        return run.info.run_id

    def finish(self) -> None:
        mlflow.end_run()
