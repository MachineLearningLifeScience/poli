"""Dockstring objective factory and function.

This module implements and registers a black box objective function
(and a repository) for dockstring [1], which is a simple API for
assessing the docking score of a small molecule to a given protein.

References
----------
[1] “DOCKSTRING: Easy Molecular Docking Yields Better Benchmarks for Ligand Design.”
    García-Ortegón, Miguel, Gregor N. C. Simm, Austin J. Tripp,
    José Miguel Hernández-Lobato, Andreas Bender, and Sergio Bacallado.
    Journal of Chemical Information and Modeling 62, no. 15 (August 8, 2022): 3486-3502.
    https://doi.org/10.1021/acs.jcim.1c01334.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import selfies as sf

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.chemistry.string_to_molecule import translate_smiles_to_selfies
from poli.core.util.isolation.instancing import get_inner_function
from poli.core.util.seeding import seed_python_numpy_and_torch


class DockstringBlackBox(AbstractBlackBox):
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
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ):
        """
        Initialize the dockstring black box object.

        Parameters
        ----------
        target_name : str
            The name of the target protein.
        string_representation : str
            The string representation of the molecules. Either SMILES or SELFIES.
            Default is SMILES.
        batch_size : int, optional
            The batch size for processing data, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the processing, by default False.
        num_workers : int, optional
            The number of workers to use for parallel processing, by default None.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.
        """
        assert (
            target_name is not None
        ), "Missing required keyword argument 'target_name'. "

        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        self.target_name = target_name
        self.string_representation = string_representation
        self.force_isolation = force_isolation

        # Testing whether we can create the isolated function
        # without raising an exception.
        _ = get_inner_function(
            isolated_function_name="dockstring__isolated",
            class_name="IsolatedDockstringFunction",
            module_to_import="poli.objective_repository.dockstring.isolated_function",
            force_isolation=force_isolation,
            target_name=target_name,
            string_representation=string_representation,
        )

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
        inner_function = get_inner_function(
            isolated_function_name="dockstring__isolated",
            class_name="IsolatedDockstringFunction",
            module_to_import="poli.objective_repository.dockstring.isolated_function",
            force_isolation=self.force_isolation,
            quiet=True,
            target_name=self.target_name,
            string_representation=self.string_representation,
        )
        return inner_function(x, context=context)

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="dockstring",
            max_sequence_length=np.inf,
            aligned=False,
            fixed_length=False,
            deterministic=True,
            alphabet=None,  # TODO: fix when we have a smiles alphabet
            log_transform_recommended=False,
            discrete=True,
            padding_token="",
        )


class DockstringProblemFactory(AbstractProblemFactory):
    """Problem factory for the Dockstring problem.

    Dockstring is a simple API for assessing the docking score
    of a small molecule to a given protein [1].

    Methods
    -------
    create(...)
        Creates a problem instance with the specified parameters.

    References
    ----------
    [1] “DOCKSTRING: Easy Molecular Docking Yields Better Benchmarks for Ligand Design.”
        García-Ortegón, Miguel, Gregor N. C. Simm, Austin J. Tripp,
        José Miguel Hernández-Lobato, Andreas Bender, and Sergio Bacallado.
        Journal of Chemical Information and Modeling 62, no. 15 (August 8, 2022): 3486-3502.
        https://doi.org/10.1021/acs.jcim.1c01334.
    """

    def create(
        self,
        target_name: str,
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        """Creates a dockstring black box function and initial observations.

        Parameters
        ----------
        target_name : str
            The name of the target protein (see dockstring for more details).
        string_representation : str, optional
            The string representation of the molecules. Either SMILES or SELFIES.
            Default is SMILES.
        seed : int, optional
            The seed value for random number generation, by default None.
        batch_size : int, optional
            The batch size for processing data, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the processing, by default False.
        num_workers : int, optional
            The number of workers to use for parallel processing, by default we
            use half the number of available CPUs (rounded down).
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.

        Returns
        -------
        problem: Problem
            A problem definition with the dockstring black box function and initial observations.
        """
        assert (
            target_name is not None
        ), "Missing required keyword argument 'target_name'. "

        if seed is not None:
            seed_python_numpy_and_torch(seed)

        dockstring_black_box = DockstringBlackBox(
            target_name=target_name,
            string_representation=string_representation,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
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

        dockstring_problem = Problem(black_box=dockstring_black_box, x0=x0)

        return dockstring_problem
