"""Implement a wrapper around the Therapeutics Data Commons (TDC) oracles [1].

When run, this script registers the TDCIsolatedFunction class as an isolated function
with the name "tdc__isolated".

References
----------
[1] “Artificial Intelligence Foundation for Therapeutic Science.”
    Huang, Kexin, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf Roohani, Jure Leskovec,
    Connor W. Coley, Cao Xiao, Jimeng Sun, and Marinka Zitnik.  Nature Chemical Biology 18, no. 10
    (October 2022): 1033-36. https://doi.org/10.1038/s41589-022-01131-2.
"""

import numpy as np

from tdc import Oracle

from poli.core.abstract_isolated_function import AbstractIsolatedFunction

from poli.core.util.chemistry.string_to_molecule import translate_selfies_to_smiles


class TDCIsolatedFunction(AbstractIsolatedFunction):
    """
    TDCBlackBox is a class that represents a black box for the
    TDC (Therapeutics Data Commons) problems.
    It inherits from the AbstractBlackBox class.

    Parameters
    -----------
    oracle_name : str
        The name of the oracle used for computing the docking score.
    from_smiles : bool, optional
        Flag indicating whether the input molecules are in SMILES format. Defaults to True.

    Attributes
    ----------
    oracle : Oracle
        An instance of the Oracle class from TDC.
    from_smiles : bool
        Flag indicating whether the input molecules are in SMILES format.
    """

    def __init__(
        self,
        oracle_name: str,
        from_smiles: bool = True,
        **kwargs_for_oracle,
    ):
        """
        Initialize the TDCIsolatedFunction class.

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
        **kwargs_for_oracle : dict, optional
            Additional keyword arguments for the oracle.
        """
        super().__init__()
        self.oracle = Oracle(name=oracle_name, **kwargs_for_oracle)
        self.from_smiles = from_smiles

    def __call__(self, x, context=None):
        """
        Assuming x is an array of strings,
        we concatenate them and then
        compute the oracle score.

        Parameters
        -----------
        x : array-like
            An array of strings representing the input molecules.
        context : any, optional
            Additional context information. Defaults to None.

        Returns
        --------
        scores : array-like
            An array of oracle scores computed for each input molecule.
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


if __name__ == "__main__":
    from poli.core.registry import register_isolated_function

    register_isolated_function(
        TDCIsolatedFunction,
        name="tdc__isolated",
        conda_environment_name="poli__tdc",
    )
