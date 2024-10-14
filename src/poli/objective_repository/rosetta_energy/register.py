from ctypes import Union
from pathlib import Path
from typing import Callable, List

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.isolation.instancing import get_inner_function, instance_function_as_isolated_process
from poli.objective_repository.rosetta_energy.information import (
    rosetta_energy_information,
)


CONSENT_FILE = Path(__file__).parent.resolve() / ".pyrosetta_accept.txt" # TODO: stores under poli repo, should this be a hidden file under home?


def has_opted_in(consent_file: Path = CONSENT_FILE) -> bool:
    if consent_file.exists():
        with open(consent_file, "r") as file:
            consent_status = file.read().strip()
            return consent_status == "accepted"
    return False


def opt_in_wrapper(f: Callable, *args, **kwargs):
    if not has_opted_in():
        agreement = input("I have read and accept the License Agreements of PyRosetta, subject to the Rosetta™ license. ([Y]es/[N]o) \n See https://www.pyrosetta.org/home/licensing-pyrosetta and https://els2.comotion.uw.edu/product/rosetta .")
        if agreement.strip().lower() == "yes" or agreement.strip().lower() == "y":
            with open(CONSENT_FILE, "w") as file:
                file.write("accepted")
            return f
        else:
            print("You must accept and be in compliance with the original PyRosetta, Rosetta™ license.")
            raise RuntimeError
    else:
        return f


class RosettaEnergyBlackBox(AbstractBlackBox):
    def __init__(
        self,
        wildtype_pdb_path: Union[Path, List[Path]],
        your_second_arg: List[float], # TODO: account for PyRosetta features here
        your_kwarg: str = ...,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        self.force_isolation = force_isolation

        # ... your manipulation of args and kwargs.

        # Importing the isolated logic if we can:
        f = get_inner_function(
            isolated_function_name="rosetta__isolated",
            class_name="RosettaEnergyIsolatedLogic",
            module_to_import="poli.objective_repository.rosetta_energy.isolated_function",
            force_isolation=self.force_isolation
        )
        self.inner_function = opt_in_wrapper(f)


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
        evaluation_budget: int = float("inf"),
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
