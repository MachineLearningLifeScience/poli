from typing import Tuple, List
import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem

from poli.core.util.isolation.instancing import instance_function_as_isolated_process
from poli.objective_repository.rosetta_energy.information import rosetta_energy_information


class RosettaEnergyBlackBox(AbstractBlackBox):
    def __init__(
            self,
            your_arg: str,
            your_second_arg: List[float],
            your_kwarg: str = ...,
            batch_size: int = None,
            parallelize: bool = False,
            num_workers: int = None,
            evaluation_budget: int = float("inf")
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

        # ... your manipulation of args and kwargs.

        # Importing the isolated logic if we can:
        try:
            #from poli.objective_repository.rosetta_energy.isolated_function import

            self.inner_function = None
        except ImportError:
            # If we weren't able to import it, we can still
            # create it in an isolated process:
            self.inner_function = instance_function_as_isolated_process(
                name=self.info.get_problem_name()  # The same name in `isolated_function.py`.
            )

    # Boilerplate for the black box call:
    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        return self.inner_function(x, context)

    # A static method that gives you access to the information.
    @staticmethod
    def get_black_box_info() -> BlackBoxInformation:
        return rosetta_energy_information


class RosettaEnergyProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> BlackBoxInformation:
        return rosetta_energy_information

    def create(
            self,
            seed: int = None,
            your_arg: str = ...,
            your_second_arg: List[float] = ...,
            your_kwarg: str = ...,
            batch_size: int = None,
            parallelize: bool = False,
            num_workers: int = None,
            evaluation_budget: int = float("inf")
    ) -> Problem:
        # Manipulate args and kwargs you might need at creation time...
        ...

        # Creating your black box function
        f = RosettaEnergyBlackBox(
            your_arg=your_arg,
            your_second_arg=your_second_arg,
            your_kwarg=your_kwarg,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

        # Your first input (an np.array[str] of shape [b, L] or [b,])
        x0 = ...

        return Problem(f, x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem_from_repository

    blackbox = RosettaEnergyBlackBox(your_arg=None, your_second_arg=None)

    # Once we have created a simple conda enviroment
    # (see the environment.yml file in this folder),
    # we can register our problem s.t. it uses
    # said conda environment.
    register_problem_from_repository(
        name=blackbox.get_black_box_info().get_problem_name()
    )
