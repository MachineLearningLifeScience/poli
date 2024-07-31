from typing import Literal

import numpy as np
from dockstring import load_target

from poli.core.abstract_isolated_function import AbstractIsolatedFunction
from poli.core.registry import register_isolated_function
from poli.core.util.chemistry.string_to_molecule import translate_selfies_to_smiles


class IsolatedDockstringFunction(AbstractIsolatedFunction):
    """
    Black box implementation for the Dockstring problem.

    Dockstring is a simple API for assessing the docking score
    of a small molecule to a given protein [1].

    Parameters
    ----------
    target_name : str
        The name of the target protein.
    string_representation : str, optional
        The string representation of the molecules. Either SMILES or SELFIES.
        Default is SMILES.
    batch_size : int, optional
        The batch size for processing multiple inputs simultaneously, by default None.
    parallelize : bool, optional
        Flag indicating whether to parallelize the computation, by default False.
    num_workers : int, optional
        The number of workers to use for parallel computation, by default None.
    evaluation_budget:  int, optional
        The maximum number of function evaluations. Default is infinity.

    Attributes
    ----------
    alphabet : dict
        The mapping of symbols to their corresponding indices in the alphabet.

    Methods
    -------
    _black_box(x, context=None)
        The black box function.

    References
    ----------
    [1] “DOCKSTRING: Easy Molecular Docking Yields Better Benchmarks for Ligand Design.”
        García-Ortegón, Miguel, Gregor N. C. Simm, Austin J. Tripp,
        José Miguel Hernández-Lobato, Andreas Bender, and Sergio Bacallado.
        Journal of Chemical Information and Modeling 62, no. 15 (August 8, 2022): 3486-3502.
        https://doi.org/10.1021/acs.jcim.1c01334.
    """

    def __init__(
        self,
        target_name: str,
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
    ):
        """
        Initialize the dockstring black box object.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for processing data, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the processing, by default False.
        num_workers : int, optional
            The number of workers to use for parallel processing, by default None.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.
        target_name : str
            The name of the target protein.
        string_representation : str
            The string representation of the molecules. Either SMILES or SELFIES.
            Default is SMILES.
        """
        assert (
            target_name is not None
        ), "Missing required keyword argument 'target_name'. "

        self.target_name = target_name
        self.string_representation = string_representation

        self.target = load_target(target_name)

    def __call__(self, x: np.ndarray, context=None) -> np.ndarray:
        """Evaluating the black box function.

        Parameters
        ----------
        x : np.ndarray
            A molecule represented as a string of shape [b, L],
            where b is the batch size and L is the length of the string.
            We expect the elements in a row of x to be the tokens
            of a molecule string representation.
        context : any, optional
            Additional context information for the evaluation. Defaults to None.

        Returns
        -------
        y: np.ndarray
            The output of the black box function.

        Raises
        ------
        AssertionError
            If the input is not a 2D array of strings.
        Exception
            If the docking score cannot be computed.
        """
        assert len(x.shape) == 2, "Expected a 2D array of strings. "
        molecules_as_strings = ["".join(x_i) for x_i in x]

        if self.string_representation == "SELFIES":
            molecules_as_smiles = translate_selfies_to_smiles(molecules_as_strings)
        else:
            molecules_as_smiles = molecules_as_strings

        # Parallelization is handled by the __call__ method of
        # the AbstractBlackBox class.
        scores = []
        for smiles in molecules_as_smiles:
            try:
                score = self.target.dock(smiles)[0]
            except Exception:
                score = np.nan
            scores.append(score)

        # Since our goal is maximization, and scores in dockstring
        # are better if they are lower, we return the negative of
        # the scores.
        return -np.array(scores).reshape(-1, 1)


if __name__ == "__main__":
    # One example of loading up this black box:
    # inner_dockstring_black_box = IsolatedDockstringFunction(
    #     target_name="abl1",
    #     string_representation="SMILES",
    # )

    # TODO: maybe force the user to instanciate.
    register_isolated_function(
        IsolatedDockstringFunction,
        name="dockstring__isolated",
        conda_environment_name="poli__dockstring",
    )
