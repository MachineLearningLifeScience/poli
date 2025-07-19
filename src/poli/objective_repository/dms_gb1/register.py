"""
This module implements the DMS GB1 black-box,
using the data available from [1].

The black-box is a lookup of the combinatorically complete mutation landscape.
The proposed sequence has to exist and the associated value is returned.

The dataset reference is [2].

[1] "Active Learning-Assisted Directed Evolution"
Jason Yang, Ravi G. Lal, James C. Bowden, Raul Astudillo, Mikhail A. Hameedi, Sukhvinder Kaur, Matthew Hill, Yisong Yue, Frances H. Arnold
bioRxiv 2024.07.27.605457; doi: https://doi.org/10.1101/2024.07.27.605457.
[2] "Learning protein fitness landscapes with deep mutational scanning data from multiple sources"
Lin Chen, Zehong Zhang, Zhenghao Li, Rui Li, Ruifeng Huo, Lifan Chen, Dingyan Wang, Xiaomin Luo, Kaixian Chen, Cangsong Liao, Mingyue Zheng,
Cell Systems,
Volume 14, Issue 8, 2023,
ISSN 2405-4712; doi: https://doi.org/10.1016/j.cels.2023.07.003.]

All rights of the data remain with the authors of the respective publications ([1], [2])
under the licenses of the provided resources.

"""

from __future__ import annotations

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.problem import Problem
from poli.core.util.isolation.instancing import get_inner_function
from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli.core.util.seeding import seed_python_numpy_and_torch


class DMSGB1BlackBox(AbstractBlackBox):
    """
    GB1 Black Box implementation.

    Parameters
    ----------
    negative: bool, optional
        Apply sign-flip on observations for a minimizing black-box.
    experiment_id : str, optional
        The experiment ID, by default None.
    batch_size : int, optional
        The batch size for parallel evaluation, by default None.
    parallelize : bool, optional
        Flag indicating whether to parallelize evaluation, by default False.
    num_workers : int, optional
        The number of workers for parallel evaluation, by default None.
    evaluation_budget : int, optional
        The evaluation budget, by default None).

    Methods
    -------
    _black_box(x, context=None)
        The main black box method, i.e.
        check x exists and return the DMS value.
    _load_dms_data()
        This function loads the datasets, found under assets.

    Notes
    -----
    ...
    """

    def __init__(
        self,
        negative: bool = False,
        experiment_id: str | None = None,
        batch_size: int | None = None,
        parallelize: bool = False,
        num_workers: int | None = None,
        evaluation_budget: int | None = None,
        force_isolation: bool = False,
    ):
        """
        Initialize the DMSGB1 Register object.

        Parameters:
        -----------
        negative: bool, optional
            Apply sign flip to observations, ie. for minimizing black-box.
        experiment_id : str, optional
            The experiment ID, by default None.
        batch_size : int, optional
            The batch size for parallel evaluation, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize evaluation, by default False.
        num_workers : int, optional
            The number of workers for parallel evaluation, by default None.
        evaluation_budget : int, optional
            The evaluation budget, by default None).
        """
        if parallelize:
            print(
                "poli 🧪: BlackBox parallelization is handled by the isolated logic. Disabling it."
            )
            parallelize = False
        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
        self.negative = negative
        self.force_isolation = force_isolation
        self.experiment_id = experiment_id
        self.inner_function = get_inner_function(
            isolated_function_name="dms_gb1__isolated",
            class_name="DMSGB1IsolatedLogic",
            module_to_import="poli.objective_repository.dms_gb1.isolated_function",
            force_isolation=self.force_isolation,
            experiment_id=self.experiment_id,
        )
        self.x0 = self.inner_function.x0

    def _black_box(self, x, context=None):
        """
        Computes the fitness of the mutant(s) in x.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape [b, L] containing strings.
        context : dict, optional
            Additional context information (default is None).

        Returns
        -------
        y : np.ndarray
            The fitness of the mutant(s) in x.

        Notes
        -----
        - x is a np.array[str] of shape [b, L], where L is the length
          of the chain of residues, and b is the batch size.
          We process it by concantenating the array into a single string,
          where we assume the padding to be an empty string (if there was any).
          Each of these x_i's will be matched to the wildtype in self.  wildtype_residue_strings with the lowest Hamming distance.
        - negative flag inverts sign of observations, ie. for minimizing the black-box
        """
        y = self.inner_function(x, context=context)
        if self.negative:
            y = -y
        return y

    def get_black_box_info(self) -> BlackBoxInformation:
        """
        Returns the black box information for DMSGB1.
        """
        is_aligned = True
        is_fixed_length = True
        max_sequence_length = max([len("".join(x)) for x in self.x0])
        return BlackBoxInformation(
            name="dms_gb1",
            max_sequence_length=max_sequence_length,
            aligned=is_aligned,
            fixed_length=is_fixed_length,
            deterministic=True,
            alphabet=AMINO_ACIDS,
            log_transform_recommended=False,
            discrete=True,
            fidelity="low",
            padding_token="",
        )


class DMSGB1ProblemFactory(AbstractProblemFactory):
    def create(
        self,
        negative: bool = False,
        experiment_id: str | None = None,
        seed: int | None = None,
        batch_size: int | None = None,
        parallelize: bool = False,
        num_workers: int | None = None,
        evaluation_budget: int | None = None,
        force_isolation: bool = False,
    ) -> Problem:
        """
        Creates a DMS GB1 black box instance, alongside initial
        observations.

        Parameters
        ----------
        negative: bool, optional
            Invert signs of observations, by default False.
        experiment_id : str, optional
            The experiment ID, by default None.
        seed : int, optional
            The seed value for random number generation, by default None.
        batch_size : int, optional
            The batch size for parallel evaluation, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize evaluation, by default False.
        num_workers : int, optional
            The number of workers for parallel evaluation, by default None.
        evaluation_budget : int, optional
            The evaluation budget, by default None).

        Returns
        -------
        f : DMSGB1BlackBox
            The DMS black box instance.
        x0 : np.ndarray
            The initial observations (i.e. the wildtype as a sequence
            of amino acids).
        y0 : np.ndarray
            The initial observations (i.e. the stability of the wildtypes).
        """
        if seed is not None:
            seed_python_numpy_and_torch(seed)

        f = DMSGB1BlackBox(
            negative=negative,
            experiment_id=experiment_id,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )

        # Constructing x0
        # (Moved to the isolated logic)
        x0 = f.inner_function.x0

        problem = Problem(f, x0)

        return problem
