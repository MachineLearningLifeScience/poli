"""In this module, we implement a synthetic-accessibility 
objective using the TDC oracles [1].

[1] Huang, Kexin, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf Roohani, Jure Leskovec, Connor W. Coley, Cao Xiao, Jimeng Sun, and Marinka Zitnik. “Artificial Intelligence Foundation for Therapeutic Science.” Nature Chemical Biology 18, no. 10 (October 2022): 1033-36. https://doi.org/10.1038/s41589-022-01131-2.

See also
-------
The website for TDC: https://tdcommons.ai/
"""

import numpy as np

from tdc import Oracle

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.chemistry.string_to_molecule import translate_selfies_to_smiles


class TDCBlackBox(AbstractBlackBox):
    """
    TDCBlackBox is a class that represents a black box for the
    TDC (Therapeutics Data Commons) problems.
    It inherits from the AbstractBlackBox class.

    Parameters:
    -----------
    oracle_name : str
        The name of the oracle used for computing the docking score.
    info : ProblemSetupInformation
        An instance of the ProblemSetupInformation class that contains information about the problem setup.
    batch_size : int, optional
        The batch size for processing multiple inputs in parallel. Defaults to None.
    parallelize : bool, optional
        Flag indicating whether to parallelize the computation. Defaults to False.
    num_workers : int, optional
        The number of workers to use for parallel computation. Defaults to None.
    evaluation_budget : int, optional
        The evaluation budget. Defaults to infinity.
    from_smiles : bool, optional
        Flag indicating whether the input molecules are in SMILES format. Defaults to True.

    Attributes:
    -----------
    oracle : Oracle
        An instance of the Oracle class from TDC.
    from_smiles : bool
        Flag indicating whether the input molecules are in SMILES format.
    """

    def __init__(
        self,
        oracle_name: str,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        from_smiles: bool = True,
    ):
        """
        Initialize the TDCBlackBox class.

        Parameters
        ----------
        oracle_name : str
            The name of the oracle.
        info : ProblemSetupInformation
            The problem setup information.
        batch_size : int, optional
            The batch size, by default None.
        parallelize : bool, optional
            Whether to parallelize the computation, by default False.
        num_workers : int, optional
            The number of workers to use for parallel computation, by default None.
        from_smiles : bool, optional
            Whether to use SMILES representation, by default True.
        """
        super().__init__(info, batch_size, parallelize, num_workers, evaluation_budget)
        self.oracle = Oracle(name=oracle_name)
        self.from_smiles = from_smiles

    def _black_box(self, x, context=None):
        """
        Assuming x is an array of strings,
        we concatenate them and then
        compute the docking score.

        Parameters:
        -----------
        x : array-like
            An array of strings representing the input molecules.
        context : any, optional
            Additional context information. Defaults to None.

        Returns:
        --------
        scores : array-like
            An array of docking scores computed for each input molecule.
        """
        if not x.dtype.kind in ["U", "S"]:
            raise ValueError(
                f"We expect x to be an array of strings, but we got {x.dtype}"
            )

        molecule_strings = ["".join([x_ij for x_ij in x_i.flatten()]) for x_i in x]

        if not self.from_smiles:
            molecule_strings = translate_selfies_to_smiles(molecule_strings)

        scores = []
        for molecule_string in molecule_strings:
            scores.append(self.oracle(molecule_string))

        return np.array(scores).reshape(-1, 1)
