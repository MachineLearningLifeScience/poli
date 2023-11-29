"""
This module implements and registers a black box
objective function (and a repository) for dockstring [1].

[1] García-Ortegón, Miguel, Gregor N. C. Simm, Austin J. Tripp,
    José Miguel Hernández-Lobato, Andreas Bender, and Sergio Bacallado.
    “DOCKSTRING: Easy Molecular Docking Yields Better Benchmarks for Ligand Design.”
    Journal of Chemical Information and Modeling 62, no. 15 (August 8, 2022): 3486-3502.
    https://doi.org/10.1021/acs.jcim.1c01334.

"""
from typing import Tuple

import numpy as np

import selfies as sf

from dockstring import load_target

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.chemistry.string_to_molecule import (
    translate_selfies_to_smiles,
    translate_smiles_to_selfies,
)

from poli.core.util.seeding import seed_numpy, seed_python


class DockstringBlackBox(AbstractBlackBox):
    """
    Black box implementation for the Dockstring problem.

    Dockstring is a simple API for assessing the docking score
    of a small molecule to a given protein [1].

    Parameters
    ----------
    info : ProblemSetupInformation
        The problem setup information.
    batch_size : int, optional
        The batch size for processing multiple inputs simultaneously, by default None.
    parallelize : bool, optional
        Flag indicating whether to parallelize the computation, by default False.
    num_workers : int, optional
        The number of workers to use for parallel computation, by default None.
    target_name : str
        The name of the target protein.
    string_representation : str
        The string representation of the molecules. Either SMILES or SELFIES.
        Default is SMILES.

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
    [1] García-Ortegón, Miguel, Gregor N. C. Simm, Austin J. Tripp,
        José Miguel Hernández-Lobato, Andreas Bender, and Sergio Bacallado.
        “DOCKSTRING: Easy Molecular Docking Yields Better Benchmarks for Ligand Design.”
        Journal of Chemical Information and Modeling 62, no. 15 (August 8, 2022): 3486-3502.
        https://doi.org/10.1021/acs.jcim.1c01334.
    """

    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        target_name: str = None,
        string_representation: str = "SMILES",
    ):
        """
        Initialize the dockstring black box object.

        Parameters
        ----------
        info : ProblemSetupInformation
            The problem setup information object.
        batch_size : int, optional
            The batch size for processing data, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the processing, by default False.
        num_workers : int, optional
            The number of workers to use for parallel processing, by default None.
        target_name : str
            The name of the target protein.
        string_representation : str
            The string representation of the molecules. Either SMILES or SELFIES.
            Default is SMILES.
        """
        assert (
            target_name is not None
        ), "Missing required keyword argument 'target_name'. "

        super().__init__(
            info=info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
        )
        self.target_name = target_name
        self.string_representation = string_representation

        self.target = load_target(target_name)

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
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

        # TODO: Should we parallelize?
        scores = []
        for smiles in molecules_as_smiles:
            try:
                score = self.target.dock(smiles)[0]
            except Exception as e:
                score = np.nan
            scores.append(score)

        # Since our goal is maximization, and scores in dockstring
        # are better if they are lower, we return the negative of
        # the scores.
        return -np.array(scores).reshape(-1, 1)


class DockstringProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        # TODO: We might change this in the future for a
        # default dictionary, depending on whether we
        # are using SMILES or SELFIES.
        alphabet = None

        return ProblemSetupInformation(
            name="dockstring",
            max_sequence_length=np.inf,
            aligned=False,
            alphabet=alphabet,
        )

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        target_name: str = None,
        string_representation: str = "SMILES",
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        assert (
            target_name is not None
        ), "Missing required keyword argument 'target_name'. "

        seed_numpy(seed)
        seed_python(seed)

        dockstring_black_box = DockstringBlackBox(
            info=self.get_setup_information(),
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            target_name=target_name,
            string_representation=string_representation,
        )

        # Using the initial example they provide in the
        # dockstring documentation (i.e. DRD2 and Risperidone)
        risperidone_smiles = "CC1=C(C(=O)N2CCCCC2=N1)CCN3CCC(CC3)C4=NOC5=C4C=CC(=C5)F"
        if string_representation.upper() == "SMILES":
            # TODO: replace for proper smiles tokenization.
            x0 = np.array([list(risperidone_smiles)])
        elif string_representation.upper() == "SELFIES":
            risperidone_selfies = translate_smiles_to_selfies([risperidone_smiles])[0]
            risperidone_selfies_as_tokens = list(sf.split_selfies(risperidone_selfies))
            x0 = np.array([risperidone_selfies_as_tokens])
        else:
            raise ValueError(
                f"Invalid string representation. Expected SMILES or SELFIES but received {string_representation}."
            )

        return dockstring_black_box, x0, dockstring_black_box(x0)


if __name__ == "__main__":
    from poli.core.registry import register_problem

    dockstring_problem_factory = DockstringProblemFactory()
    register_problem(
        dockstring_problem_factory,
        conda_environment_name="poli__dockstring",
    )
