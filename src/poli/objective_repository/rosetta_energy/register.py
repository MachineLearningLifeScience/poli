from __future__ import annotations

from pathlib import Path
from typing import Callable, List

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.isolation.instancing import (
    get_inner_function,
    instance_function_as_isolated_process,
)
from poli.objective_repository.rosetta_energy.information import (
    rosetta_energy_information,
)

CONSENT_FILE = Path(__file__).parent.resolve() / ".pyrosetta_accept.txt"


def has_opted_in(consent_file: Path = CONSENT_FILE) -> bool:
    if consent_file.exists():
        with open(consent_file, "r") as file:
            consent_status = file.read().strip()
            return consent_status == "accepted"
    return False


def opt_in_wrapper(f: Callable, *args, **kwargs):
    if not has_opted_in():
        agreement = input(
            "I have read and accept the License Agreements of PyRosetta, subject to the Rosetta™ license. ([Y]es/[N]o) \n See https://www.pyrosetta.org/home/licensing-pyrosetta and https://els2.comotion.uw.edu/product/rosetta ."
        )
        if agreement.strip().lower() == "yes" or agreement.strip().lower() == "y":
            with open(CONSENT_FILE, "w") as file:
                file.write("accepted")
            return f
        else:
            print(
                "You must accept and be in compliance with the original PyRosetta, Rosetta™ license."
            )
            raise RuntimeError
    else:
        return f


class RosettaEnergyBlackBox(AbstractBlackBox):
    def __init__(
        self,
        wildtype_pdb_path: Path | List[Path],
        score_function: str = "default",
        seed: int = 0,
        unit: str = "DDG",
        conversion_factor: float = 2.9,
        clean: bool = True,
        relax: bool = True,
        pack: bool = True,
        cycle: int = 3,
        n_threads: int = 4,
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
        assert wildtype_pdb_path is not None
        self.force_isolation = force_isolation
        self.wildtype_pdb_path = wildtype_pdb_path
        self.score_function = score_function
        self.seed = seed
        self.unit = unit
        self.conversion_factor = conversion_factor
        self.clean = clean
        self.relax = relax
        self.pack = pack
        self.cycle = cycle
        self.n_threads = n_threads

        try:

            inner_function = get_inner_function(
                isolated_function_name="rosetta_energy__isolated",
                class_name="RosettaEnergyIsolatedLogic",
                module_to_import="poli.objective_repository.rosetta_energy.isolated_function",
                force_isolation=self.force_isolation,
                wildtype_pdb_path=self.wildtype_pdb_path,
                score_function=self.score_function,
                seed=self.seed,
                unit=self.unit,
                conversion_factor=self.conversion_factor,
                clean=self.clean,
                relax=self.relax,
                pack=self.pack,
                cycle=self.cycle,
                n_threads=self.n_threads,
            )
            self.inner_function = opt_in_wrapper(inner_function)
            self.x0 = self.inner_function.x0

        except ImportError:
            # If we weren't able to import it, we can still
            # create it in an isolated process:
            self.inner_function = instance_function_as_isolated_process(
                name=self.info.get_problem_name()  # The same name in `isolated_function.py`.
            )

    def _black_box(self, x: np.ndarray, context: dict = None) -> np.ndarray:
        """
        Computes the stability of the mutant(s) in x.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape [b, L] containing strings.
        context : dict, optional
            Additional context information (default is None).

        Returns
        -------
        y : np.ndarray
            The stability(/REUs) of the mutant(s) in x.
        """
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
        wildtype_pdb_path: Path | List[Path],
        score_function: str = "default",
        seed: int = 0,
        unit: str = "DDG",
        conversion_factor: float = 2.9,
        clean: bool = True,
        relax: bool = True,
        pack: bool = True,
        cycle: int = 3,
        n_threads: int = 4,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        # Creating your black box function
        f = RosettaEnergyBlackBox(
            wildtype_pdb_path=wildtype_pdb_path,
            score_function=score_function,
            seed=seed,
            unit=unit,
            conversion_factor=conversion_factor,
            clean=clean,
            relax=relax,
            pack=pack,
            cycle=cycle,
            n_threads=n_threads,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )

        # Your first input (an np.array[str] of shape [b, L] or [b,])
        x0 = f.inner_function.x0

        return Problem(f, x0)
