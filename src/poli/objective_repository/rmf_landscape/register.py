"""
Implements a (tunable) fitness landscape of the type RMF [1] for NA [1], AA [2] inputs.

References
----------
[1] Adaptation in Tunably Rugged Fitness Landscapes: The Rough Mount Fuji Model.
    Neidhart J., Szendro I.G., and Krug, J. Genetics 198, 699-721 (2014). https://doi.org/10.1534/genetics.114.167668
[2] Analysis of a local fitness landscape with a model of the rough Mt. Fuji-type landscape: Application to prolyl endopeptidase and thermolysin.
    Aita T., Uchiyama H., et al. Biopolymers 54, 64-79 (2000). https://doi.org/10.1002/(SICI)1097-0282(200007)54:1<64::AID-BIP70>3.0.CO;2-R
"""

from __future__ import annotations

from typing import List

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.isolation.instancing import get_inner_function
from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli.core.util.seeding import seed_python_numpy_and_torch


class RMFBlackBox(AbstractBlackBox):
    """
    RMF Black Box implementation.

    Parameters
    ----------
    wildtype : str
        The wildtype amino-acid sequence (aka reference sequence) against which all RMF values are computed against.
    wt_val : float , optional
        The reference value for the WT, zero if observations are standardized, else float value e.g. ddGs
    c : float, optional
        Constant scalar used in RMF computation, by default is the normalizing constant relative to alphabet size
    kappa : float, optional
        Parameterizes the generalized Pareto distribution, by default 0.1 .
        Determines what type of distribution will be sampled from exponential family, Weibull, etc.
    seed : int, optional
        Random seed for replicability of results, by default None.
    alphabet : List[str], optional
        Type of alphabet of the sequences, by default Amino Acids.
        Nucleic Acids possible.
    batch_size : int, optional
        The batch size for parallel evaluation, by default None.
    parallelize : bool, optional
        Flag to parallelize evaluation, by default False.
    num_workers : int, optional
        The number of workers for parallel evaluation, by default None.
    evaluation_budget : int, optional
        The evaluation budget, by default float("inf").
    force_isolation : bool, optional
        Run in an isolated environment and process, by default False.
    """

    def __init__(
        self,
        wildtype: str,
        wt_val: float = 0.0,
        c: float | None = None,
        kappa: float = 0.1,
        seed: int | None = None,
        alphabet: List[str] | None = None,
        batch_size: int | None = None,
        parallelize: bool | None = False,
        num_workers: int | None = None,
        evaluation_budget: int | None = float("inf"),
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
        _ = get_inner_function(  # NOTE: this implicitly registers
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

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="rmf_landscape",
            max_sequence_length=np.inf,
            aligned=True,
            fixed_length=True,
            deterministic=False,
            alphabet=AMINO_ACIDS,  # TODO: differentiate between AA and NA inputs?
            log_transform_recommended=False,
            discrete=True,
        )


class RMFProblemFactory(AbstractProblemFactory):
    """
    Problem factory for the rough Mt Fuji model problem.

    Methods
    -------
    create(...)
        Creates RMF problem instance with specified parameters.
    """

    def create(
        self,
        wildtype: List[str] | str,
        wt_val: float | None = 0.0,
        c: float | None = None,
        kappa: float = 0.1,
        alphabet: List[str] | None = None,
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
