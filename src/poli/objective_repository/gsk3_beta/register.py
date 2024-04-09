"""
Implements the GSK3Beta task using the TDC oracles [1].

References
----------
[1] Artificial intelligence foundation for therapeutic science.
    Huang, K., Fu, T., Gao, W. et al.  Nat Chem Biol 18, 1033-1036 (2022).
    https://doi.org/10.1038/s41589-022-01131-2
"""

from typing import Literal

import numpy as np

import selfies as sf


from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.problem import Problem

from poli.core.util.isolation.instancing import instance_function_as_isolated_process

from poli.core.util.chemistry.string_to_molecule import translate_smiles_to_selfies

from poli.core.util.seeding import seed_numpy, seed_python

from poli.objective_repository.gsk3_beta.information import gsk3_beta_info


class GSK3BetaBlackBox(AbstractBlackBox):
    """
    A black box for the Glycogen Synthase Kinase 3 Beta (GSK3Beta) task,
    using the Therapeutics Data Commons' oracles [1].

    Parameters
    ----------
    string_representation : Literal["SMILES", "SELFIES"], optional
        A string (either "SMILES" or "SELFIES") specifying which
        molecule representation you plan to use.
    batch_size : int, optional
        The batch size for simultaneous execution, by default None.
    parallelize : bool, optional
        Flag indicating whether to parallelize execution, by default False.
    num_workers : int, optional
        The number of workers for parallel execution, by default None.
    evaluation_budget:  int, optional
        The maximum number of function evaluations. Default is infinity.
    force_isolation: bool, optional
        Whether to force the isolation of the black box. Default is False.

    Attributes
    ----------
    oracle_name : str
        The name of the oracle.

    Methods
    -------
    __init__(self, info, batch_size=None, parallelize=False, num_workers=None, from_smiles=True)
        Initializes a new instance of the DRD3BlackBox class.

    References
    ----------
    [1] Artificial intelligence foundation for therapeutic science.
        Huang, K., Fu, T., Gao, W. et al.  Nat Chem Biol 18, 1033-1036 (2022).
        https://doi.org/10.1038/s41589-022-01131-2
    """

    def __init__(
        self,
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
        force_isolation: bool = False,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ):
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

        if not force_isolation:
            try:
                from poli.objective_repository.gsk3_beta.isolated_function import (
                    GSK3BetaIsolatedFunction,
                )

                self.inner_function = GSK3BetaIsolatedFunction(
                    string_representation=string_representation
                )
            except ImportError:
                self.inner_function = instance_function_as_isolated_process(
                    name="gsk3_beta__isolated",
                    string_representation=string_representation,
                )
        else:
            self.inner_function = instance_function_as_isolated_process(
                name="gsk3_beta__isolated",
                string_representation=string_representation,
            )

    @staticmethod
    def get_black_box_info() -> BlackBoxInformation:
        return gsk3_beta_info

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
        return self.inner_function(x, context)


class GSK3BetaProblemFactory(AbstractProblemFactory):
    """
    A factory for creating Glycogen synthase kinase 3 beta problems.

    Methods
    ------
    get_setup_information:
        Retrieves the setup information for the problem.
    create:
        Creates a DRD3 docking problem.
    """

    def get_setup_information(self) -> BlackBoxInformation:
        """
        Retrieves the setup information for the problem.

        Returns
        --------
        problem_info: ProblemSetupInformation
            The setup information for the problem.
        """
        return gsk3_beta_info

    def create(
        self,
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        """
        Create a TDCBlackBox object for DRD3 docking.

        Parameters
        ----------
        string_representation : str, optional
            The string representation of the molecules. Must be either 'SMILES' or 'SELFIES'. Default is 'SMILES'.
        seed : int, optional
            Seed for random number generators. If None, no seed is set.
        batch_size : int, optional
            Number of molecules to process in parallel. If None, the default batch size is used.
        parallelize : bool, optional
            Whether to parallelize the docking process. Default is False.
        num_workers : int, optional
            Number of worker processes to use for parallelization. If None, the number of available CPU cores is used.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.
        force_isolation: bool, optional
            Whether to force the isolation of the black box. Default is False.

        Returns
        -------
        problem : Problem
            A problem instance containing the black box, and an initial value x0.

        Raises
        ------
        ValueError
            If the string_representation is not 'SMILES' or 'SELFIES'.
        """
        # We start by seeding the RNGs
        if seed is not None:
            seed_numpy(seed)
            seed_python(seed)

        # We check whether the string representation is valid
        if string_representation.upper() not in ["SMILES", "SELFIES"]:
            raise ValueError(
                "Missing required keyword argument: string_representation: str. "
                "String representation must be either 'SMILES' or 'SELFIES'."
            )

        f = GSK3BetaBlackBox(
            string_representation=string_representation,
            force_isolation=force_isolation,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

        # Initial example (from the TDC docs)
        x0_smiles = "CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1"
        x0_selfies = translate_smiles_to_selfies([x0_smiles])[0]

        if string_representation.upper() == "SMILES":
            x0 = np.array([list(x0_smiles)])
        else:
            x0 = np.array([list(sf.split_selfies(x0_selfies))])

        gsk3_beta_problem = Problem(
            black_box=f,
            x0=x0,
        )

        return gsk3_beta_problem


if __name__ == "__main__":
    from poli.core.registry import register_problem

    register_problem(
        GSK3BetaProblemFactory(),
        name="gs3_beta",
        conda_environment_name="poli__tdc",
        force=True,
    )
