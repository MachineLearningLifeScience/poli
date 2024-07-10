"""
Implements a (tunable) fitness landscape of the type RMF [1] for NA [1], AA [2] inputs.

References
----------
[1] Adaptation in Tunably Rugged Fitness Landscapes: The Rough Mount Fuji Model.
    Neidhart J., Szendro I.G., and Krug, J. Genetics 198, 699-721 (2014). https://doi.org/10.1534/genetics.114.167668 
[2] Analysis of a local fitness landscape with a model of the rough Mt. Fuji-type landscape: Application to prolyl endopeptidase and thermolysin.
    Aita T., Uchiyama H., et al. Biopolymers 54, 64-79 (2000). https://doi.org/10.1002/(SICI)1097-0282(200007)54:1<64::AID-BIP70>3.0.CO;2-R 
"""

from typing import List, Optional, Union

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.isolation.instancing import get_inner_function
from poli.core.util.seeding import seed_python_numpy_and_torch

from poli.core.util.isolation.instancing import instance_function_as_isolated_process

from poli.objective_repository.rmf_landscape.information import rmf_info


class RMFBlackBox(AbstractBlackBox):
    """
    TODO: docstring
    """

    def __init__(
        self,
        wildtype: str,
        wt_val: Optional[float] = 0.0,
        c: Optional[float] = None,
        kappa: Optional[float] = 0.1,
        seed: Optional[int] = None,
        alphabet: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        parallelize: Optional[bool] = False,
        num_workers: Optional[int] = None,
        evaluation_budget: Optional[int] = float("inf"),
        force_isolation: bool = False,
    ) -> None:
        """
        Initialize the RMFBlackBox object.

        Parameters
        ----------
        batch_size : int, optional
            The batch-size for parallel evaluation, default: None.
        parallelize : bool, optional
            Flag to parallelize the evaluation, default: False.
        num_workers : int, optional
            Number of workers for parallel evaluation, default: None.
        evaluation_budget : int, optional
            Maximum number of evaluations, default: float("inf").
        force_isolation: bool
            Run the blackbox in an isolated environment, default: False.
        """
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        self.wildtype = wildtype
        self.wt_val = wt_val
        self.c = c
        self.kappa = kappa
        self.alphabet = alphabet
        self.seed = seed
        self.force_isolation = force_isolation
        inner_function = get_inner_function(  # NOTE: this implicitly registers
            isolated_function_name="rmf_landscape__isolated",
            class_name="RMFIsolatedLogic",
            module_to_import="poli.objective_repository.rmf_landscape.isolated_function",
            wildtype=self.wildtype,
            wt_val=self.wt_val,
            c=self.c,
            kappa=self.kappa,
            alphabet=self.alphabet,
            seed=self.seed,
            force_isolation=self.force_isolation,
        )

    def _black_box(self, x: np.ndarray, context: None) -> np.ndarray:
        """
        Runs the given input x provided
        in the context with the RMF function and returns the
        total fitness score.

        Parameters
        -----------
        x : np.ndarray
            The input array of strings containing mutations.
        context : None
            The context for the black box computation.

        Returns
        --------
        y: np.ndarray
            The computed fitness score(s) as a numpy array.
        """
        inner_function = get_inner_function(
            isolated_function_name="rmf_landscape__isolated",
            class_name="RMFIsolatedLogic",
            module_to_import="poli.objective_repository.rmf_landscape.isolated_function",
            wildtype=self.wildtype,
            wt_val=self.wt_val,
            c=self.c,
            kappa=self.kappa,
            alphabet=self.alphabet,
            seed=self.seed,
            force_isolation=self.force_isolation,
        )
        return inner_function(x, context)

    @staticmethod
    def get_black_box_info() -> BlackBoxInformation:
        return rmf_info


class RMFProblemFactory(AbstractProblemFactory):
    """
    Problem factory for the rough Mt Fuji model problem.

    Methods
    -------
    get_setup_information()
        returns problem setup information.
    create(...)
        Creates RMF problem instance with specified parameters.
    """

    def get_setup_information(self) -> BlackBoxInformation:
        return rmf_info

    def create(
        self,
        wildtype: Union[List[str], str],
        wt_val: Optional[float] = 0.0,
        c: Optional[float] = None,
        kappa: float = 0.1,
        alphabet: Optional[List[str]] = None,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        """
        Create a RMFBlackBox object.

        Parameters
        ----------
        wildtype : List[str] | str
            Reference (wild-type) sequence is pseudo-optimum on start.
        wt_val : float, optional
            Reference function value (standardized observations) of WT.
        c : float, optional
            Constant value for function value computation.
            If None passed default value is regularizing 1/(len(alphabet)-1) .
        kappa: float
            Determines generalized Pareto continuous RV.
        alphabet: List[str], optional
            Problem alphabet used, if None is passed default: AMINO_ACIDS.
        seed : int, optional
            Seed for random number generators. If None is passed,
            the seeding doesn't take place.
        batch_size : int, optional
            Number of samples per batch for parallel computation.
        parallelize : bool, optional
            Flag indicating whether to parallelize the computation.
        num_workers : int, optional
            Number of worker processes for parallel computation.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.

        Returns
        -------
        problem : Problem
            A problem instance containing a RMFBlackBox
            function, and initial wildtypes x0.

        Raises
        ------
        ValueError
            If wildtype reference sequence is missing.
        """
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        if wildtype is None:
            raise ValueError("Missing reference sequence!")

        if isinstance(wildtype, str):
            wildtype = list(wildtype)

        f = RMFBlackBox(
            wildtype=wildtype,
            wt_val=wt_val,
            c=c,
            kappa=kappa,
            alphabet=alphabet,
            seed=seed,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )
        x0 = np.array(wildtype).reshape(1, len(wildtype))
        problem = Problem(f, x0)
        return problem


if __name__ == "__main__":
    from poli.core.registry import register_problem

    rmf_problem_factory = RMFProblemFactory()
    register_problem(
        rmf_problem_factory,
        conda_environment_name="poli__rmf",
    )
