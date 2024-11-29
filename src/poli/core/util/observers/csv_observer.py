from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time
from uuid import uuid4

import numpy as np

from poli.core.util.abstract_observer import AbstractObserver


@dataclass
class CSVObserverInitInfo:
    """Initialization information for the CSVObserver."""

    experiment_id: str
    experiment_path: str | Path = "./poli_results"


class CSVObserver(AbstractObserver):
    def initialize_observer(
        self,
        problem_setup_info: object,
        caller_info: CSVObserverInitInfo,
        seed: int,
    ) -> object:
        self.info = problem_setup_info
        self.seed = seed
        self.unique_id = f"{uuid4()}"[:8]
        self.experiment_id = caller_info.get(
            "experiment_id",
            f"{int(time())}_experiment_{problem_setup_info.name}_{seed}_{self.unique_id}",
        )
        self.experiment_path = Path(
            caller_info.get("experiment_path", "./poli_results")
        )
        self.experiment_path.mkdir(exist_ok=True, parents=True)

        if not (self.experiment_path / ".gitignore").exists():
            with open(self.experiment_path / ".gitignore", "w") as f:
                f.write("*\n")

        self.csv_file_path = self.experiment_path / f"{self.experiment_id}.csv"
        self.save_header()

    def _validate_input(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.ndim != 2:
            raise ValueError(f"x should be 2D, got {x.ndim}D instead.")
        if y.ndim != 2:
            raise ValueError(f"y should be 2D, got {y.ndim}D instead.")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x and y should have the same number of samples, got {x.shape[0]} and {y.shape[0]} respectively."
            )

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        self._validate_input(x, y)
        self.append_results(["".join(x_i) for x_i in x], [y_i for y_i in y.flatten()])

    def save_header(self):
        with open(self.csv_file_path, "w") as f:
            f.write("x,y\n")

    def append_results(self, x: list[str], y: list[float]):
        with open(self.csv_file_path, "a") as f:
            for x_i, y_i in zip(x, y):
                f.write(f"{x_i},{y_i}\n")
