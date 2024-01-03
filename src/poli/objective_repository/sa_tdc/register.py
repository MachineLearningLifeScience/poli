"""
Implements a synthetic-accessibility objective using the TDC oracles [1].

References
----------
[1] Artificial intelligence foundation for therapeutic science.
    Huang, K., Fu, T., Gao, W. et al.  Nat Chem Biol 18, 1033-1036 (2022). https://doi.org/10.1038/s41589-022-01131-2
"""
from typing import Tuple

import numpy as np

import selfies as sf

from poli.core.chemistry.tdc_black_box import TDCBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.chemistry.string_to_molecule import translate_smiles_to_selfies

from poli.core.util.seeding import seed_numpy, seed_python, seed_python_numpy_and_torch


class SABlackBox(TDCBlackBox):
    """Synthetic-accessibility black box implementation using the TDC oracles [1].

    Parameters
    ----------
    info : ProblemSetupInformation
        The problem setup information.
    batch_size : int, optional
        The batch size for simultaneous execution, by default None.
    parallelize : bool, optional
        Flag indicating whether to parallelize execution, by default False.
    num_workers : int, optional
        The number of workers for parallel execution, by default None.
    evaluation_budget:  int, optional
        The maximum number of function evaluations. Default is infinity.
    from_smiles : bool, optional
        Flag indicating whether to use SMILES strings as input, by default True.
    """

    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        from_smiles: bool = True,
    ):
        """
        Initialize the SABlackBox object.

        Parameters
        ----------
        info : ProblemSetupInformation
            The problem setup information object.
        batch_size : int, optional
            The batch size for parallel evaluation, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the evaluation, by default False.
        num_workers : int, optional
            The number of workers for parallel evaluation, by default None.
        evaluation_budget : int, optional
            The maximum number of evaluations, by default float("inf").
        from_smiles : bool, optional
            Flag indicating whether to use SMILES strings as input, by default True.
        """
        oracle_name = "SA"
        super().__init__(
            oracle_name=oracle_name,
            info=info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            from_smiles=from_smiles,
        )


class SAProblemFactory(AbstractProblemFactory):
    """Problem factory for the synthetic-accessibility problem.

    Methods
    -------
    get_setup_information()
        Returns the setup information for the problem.
    create(...)
        Creates a synthetic-accessibility problem instance with the specified parameters.
    """

    def get_setup_information(self) -> ProblemSetupInformation:
        """
        Returns the setup information for the problem.

        Returns
        --------
        problem_info: ProblemSetupInformation
            The setup information for the problem.
        """
        return ProblemSetupInformation(
            name="sa_tdc",
            max_sequence_length=np.inf,
            aligned=False,
            alphabet=None,
        )

    def create(
        self,
        string_representation: str = "SMILES",
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ) -> Tuple[SABlackBox, np.ndarray, np.ndarray]:
        """
        Creates a synthetic-accessibility problem instance with the specified parameters.

        Parameters
        -----------
        string_representation : str, optional
            The string representation of the input molecules. Default is "SMILES".
        seed:  int, optional
            The seed for random number generation. Default is None.
        batch_size:  int, optional
            The batch size for simultaneous evaluation. Default is None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the evaluation. Default is False.
        num_workers:  int, optional
            The number of workers for parallel evaluation. Default is None.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.

        Returns
        --------
        f: SABlackBox
            The synthetic-accessibility black box function.
        x0: np.ndarray
            The initial input (taken from TDC: CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1).
        y0: np.ndarray
            The initial output (i.e. the corresponding SA).
        """
        seed_numpy(seed)
        seed_python(seed)

        if string_representation.upper() not in ["SMILES", "SELFIES"]:
            raise ValueError(
                "Missing required keyword argument: string_representation: str. "
                "String representation must be either 'SMILES' or 'SELFIES'."
            )

        problem_info = self.get_setup_information()
        f = SABlackBox(
            info=problem_info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            from_smiles=string_representation.upper() == "SMILES",
        )

        # Initial example (from the TDC docs)
        x0_smiles = "CCNC(=O)c1ccc(NC(=O)N2CC[C@H](C)[C@H](O)C2)c(C)c1"
        x0_selfies = translate_smiles_to_selfies([x0_smiles])[0]

        # TODO: change for proper tokenization in the SMILES case.
        if string_representation.upper() == "SMILES":
            x0 = np.array([list(x0_smiles)])
        else:
            x0 = np.array([list(sf.split_selfies(x0_selfies))])

        return f, x0, f(x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    register_problem(
        SAProblemFactory(),
        conda_environment_name="poli__tdc",
        force=True,
    )
