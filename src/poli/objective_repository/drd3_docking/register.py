"""
This module implements the DDR3 docking task
using the TDC oracles [1].

[1] Huang, K., Fu, T., Gao, W. et al. Artificial intelligence foundation for therapeutic science. Nat Chem Biol 18, 1033-1036 (2022). https://doi.org/10.1038/s41589-022-01131-2
"""
from typing import Tuple

import numpy as np

import selfies as sf

from poli.core.chemistry.tdc_black_box import TDCBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.chemistry.string_to_molecule import translate_smiles_to_selfies

from poli.core.util.seeding import seed_numpy, seed_python


class DRD3BlackBox(TDCBlackBox):
    """
    DRD3BlackBox is a class that represents a black box for DRD3 docking.

    Parameters
    ----------
    info : ProblemSetupInformation
        The problem setup information.
    batch_size : int, optional
        The batch size for parallel execution, by default None.
    parallelize : bool, optional
        Flag indicating whether to parallelize execution, by default False.
    num_workers : int, optional
        The number of workers for parallel execution, by default None.
    from_smiles : bool, optional
        Flag indicating whether to use SMILES strings as input, by default True.

    Attributes
    ----------
    oracle_name : str
        The name of the oracle.

    Methods
    -------
    __init__(self, info, batch_size=None, parallelize=False, num_workers=None, from_smiles=True)
        Initializes a new instance of the DRD3BlackBox class.
    """

    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        from_smiles: bool = True,
    ):
        oracle_name = "3pbl_docking"
        super().__init__(
            oracle_name=oracle_name,
            info=info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            from_smiles=from_smiles,
        )


class DRD3ProblemFactory(AbstractProblemFactory):
    """
    Factory class for creating DRD3 docking problems.

    This class provides methods for creating DRD3 docking problems and retrieving setup information.

    Attributes:
    ----------
        None

    Methods:
    -------
    get_setup_information:
        Retrieves the setup information for the problem.
    create:
        Creates a DRD3 docking problem.
    """

    def get_setup_information(self) -> ProblemSetupInformation:
        """
        Retrieves the setup information for the problem.

        Returns:
        --------
        problem_info: ProblemSetupInformation
            The setup information for the problem.
        """
        return ProblemSetupInformation(
            name="drd3_docking",
            max_sequence_length=np.inf,
            aligned=False,
            alphabet=None,
        )

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        string_representation: str = "SMILES",
    ) -> Tuple[TDCBlackBox, np.ndarray, np.ndarray]:
        """
        Create a TDCBlackBox object for DRD3 docking.

        Parameters:
        ----------
        seed : int, optional
            Seed for random number generators. If None, no seed is set.
        batch_size : int, optional
            Number of molecules to process in parallel. If None, the default batch size is used.
        parallelize : bool, optional
            Whether to parallelize the docking process. Default is False.
        num_workers : int, optional
            Number of worker processes to use for parallelization. If None, the number of available CPU cores is used.
        string_representation : str, optional
            The string representation of the molecules. Must be either 'SMILES' or 'SELFIES'. Default is 'SMILES'.

        Returns:
        -------
        Tuple[TDCBlackBox, np.ndarray, np.ndarray]
            A tuple containing the TDCBlackBox object, the initial input array, and the output array.

        Raises:
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

        problem_info = self.get_setup_information()
        f = DRD3BlackBox(
            info=problem_info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            from_smiles=string_representation.upper() == "SMILES",
        )

        # Initial example (from the TDC docs)
        x0_smiles = "c1ccccc1"
        x0_selfies = translate_smiles_to_selfies([x0_smiles])[0]

        if string_representation.upper() == "SMILES":
            x0 = np.array([list(x0_smiles)])
        else:
            x0 = np.array([list(sf.split_selfies(x0_selfies))])

        return f, x0, f(x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    register_problem(
        DRD3ProblemFactory(),
        conda_environment_name="poli__lambo",
        force=True,
    )
