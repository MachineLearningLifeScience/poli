"""A simple example of how to log objective function calls using wandb.

To run this example, you will need to install wandb:

    pip install wandb
"""

import numpy as np
import wandb

from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.abstract_observer import AbstractObserver


class WandbObserver(AbstractObserver):
    def __init__(self) -> None:
        # Log into wandb
        wandb.login()

        # Some variables to keep track of the run
        self.step = 0
        self.x_table = wandb.Table(columns=["step", "x", "y"])
        super().__init__()

    def initialize_observer(
        self,
        problem_setup_info: BlackBoxInformation,
        caller_info: object,
        seed: int,
    ) -> object:
        wandb.init(
            config={
                "name": problem_setup_info.name,
                "max_sequence_length": problem_setup_info.max_sequence_length,
                "alphabet": problem_setup_info.alphabet,
                "seed": seed,
            },
        )
        return None

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        for x_i, y_i in zip(x.tolist(), y.tolist()):
            self.x_table.add_data(self.step, "".join(x_i), y_i)

        wandb.log({"table of sequences": self.x_table})
        wandb.log({"y": y}, step=self.step)

        self.step += 1

    def finish(self) -> None:
        wandb.finish()
