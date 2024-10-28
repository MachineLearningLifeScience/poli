"""
Implements the Glycogen Synthase Kinase 3 Beta (GSK3Beta) task
using the TDC oracles [1].

If you use this black box, we recommend citing [1, 2, 3, 4].

References
----------
[1] Artificial intelligence foundation for therapeutic science.
    Huang, K., Fu, T., Gao, W. et al.  Nat Chem Biol 18, 1033-1036 (2022).
    https://doi.org/10.1038/s41589-022-01131-2
[2] Multi-objective de novo drug design with conditional graph generative models.
    Li, Y., Zhang, L., Liu, Z.
    Journal of cheminformatics 10.1 (2018).
[3] Multi-objective molecule generation using interpretable substructures.
    Jin, Wengong, Regina Barzilay, and Tommi Jaakkola.
    ICML. 2020.
    https://people.csail.mit.edu/tommi/papers/JBJ_ICML2020b.pdf
[4] ExCAPE-DB: an integrated large scale dataset facilitating Big Data analysis in chemogenomics.
    Jiangming, Sun, et al.
    Journal of cheminformatics 9.1 (2017).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import selfies as sf

from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.black_box_information import BlackBoxInformation
from poli.core.chemistry.tdc_black_box import TDCBlackBox
from poli.core.chemistry.tdc_problem import TDCProblem
from poli.core.problem import Problem
from poli.core.util.chemistry.string_to_molecule import translate_smiles_to_selfies
from poli.core.util.seeding import seed_numpy, seed_python


class GSK3BetaBlackBox(TDCBlackBox):
    """
    A black box for the Glycogen Synthase Kinase 3 Beta (GSK3Beta) task,
    using the Therapeutics Data Commons' oracles [1].

    If you use this black box, we recommend citing [1, 2, 3, 4].

    Parameters
    ----------
    string_representation : Literal["SMILES", "SELFIES"], optional
        A string (either "SMILES" or "SELFIES") specifying which
        molecule representation you plan to use.
    alphabet : list[str] | None, optional
        The alphabet to be used for the SMILES or SELFIES representation.
        It is common that the alphabet depends on the dataset used, so
        it is recommended to pass it as an argument. Default is None.
    max_sequence_length : int, optional
        The maximum length of the sequence. Default is infinity.
    batch_size : int, optional
        The batch size for simultaneous execution, by default None.
    parallelize : bool, optional
        Flag indicating whether to parallelize execution, by default False.
    num_workers : int, optional
        The number of workers for parallel execution, by default None.
    evaluation_budget:  int, optional
        The maximum number of function evaluations. Default is infinity.
    force_isolation: bool, optional
        Whether to force the isolation of the black box. Default is False.

    Attributes
    ----------
    oracle_name : str
        The name of the oracle.

    Methods
    -------
    __init__(self, info, batch_size=None, parallelize=False, num_workers=None, from_smiles=True)
        Initializes a new instance of the DRD3BlackBox class.

    References
    ----------
    [1] Artificial intelligence foundation for therapeutic science.
        Huang, K., Fu, T., Gao, W. et al.  Nat Chem Biol 18, 1033-1036 (2022).
        https://doi.org/10.1038/s41589-022-01131-2
    [2] Multi-objective de novo drug design with conditional graph generative models.
        Li, Y., Zhang, L., Liu, Z.
        Journal of cheminformatics 10.1 (2018).
    [3] Multi-objective molecule generation using interpretable substructures.
        Jin, Wengong, Regina Barzilay, and Tommi Jaakkola.
        ICML. 2020.
        https://people.csail.mit.edu/tommi/papers/JBJ_ICML2020b.pdf
    [4] ExCAPE-DB: an integrated large scale dataset facilitating Big Data analysis in chemogenomics.
        Jiangming, Sun, et al.
        Journal of cheminformatics 9.1 (2017).
    """

    def __init__(
        self,
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
        alphabet: list[str] | None = None,
        max_sequence_length: int = np.inf,
        force_isolation: bool = False,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ):
        super().__init__(
            oracle_name="GSK3B",
            string_representation=string_representation,
            alphabet=alphabet,
            max_sequence_length=max_sequence_length,
            force_isolation=force_isolation,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

    def get_black_box_info(self) -> BlackBoxInformation:
        return BlackBoxInformation(
            name="gsk3_beta",
            max_sequence_length=self.max_sequence_length,
            aligned=False,
            fixed_length=False,
            deterministic=True,  # ?
            alphabet=self.alphabet,  # TODO: add alphabet once we settle for one for SMLIES/SELFIES.
            log_transform_recommended=False,
            discrete=True,
            padding_token="",
        )


class GSK3BetaProblemFactory(AbstractProblemFactory):
    """
    A factory for creating Glycogen synthase kinase 3 beta problems.

    If you use this problem factory, we recommend you cite [1, 2, 3, 4].

    Methods
    ------
    create:
        Creates a GSK3Beta docking problem.

    References
    ----------
    [1] Artificial intelligence foundation for therapeutic science.
        Huang, K., Fu, T., Gao, W. et al.  Nat Chem Biol 18, 1033-1036 (2022).
        https://doi.org/10.1038/s41589-022-01131-2
    [2] Multi-objective de novo drug design with conditional graph generative models.
        Li, Y., Zhang, L., Liu, Z.
        Journal of cheminformatics 10.1 (2018).
    [3] Multi-objective molecule generation using interpretable substructures.
        Jin, Wengong, Regina Barzilay, and Tommi Jaakkola.
        ICML. 2020.
        https://people.csail.mit.edu/tommi/papers/JBJ_ICML2020b.pdf
    [4] ExCAPE-DB: an integrated large scale dataset facilitating Big Data analysis in chemogenomics.
        Jiangming, Sun, et al.
        Journal of cheminformatics 9.1 (2017).
    """

    def create(
        self,
        string_representation: Literal["SMILES", "SELFIES"] = "SMILES",
        alphabet: list[str] | None = None,
        max_sequence_length: int = np.inf,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        """
        Creates GSK3B problems.

        Parameters
        ----------
        string_representation : str, optional
            The string representation of the molecules. Must be either 'SMILES' or 'SELFIES'. Default is 'SMILES'.
        alphabet : list[str] | None, optional
            The alphabet to be used for the SMILES or SELFIES representation.
            It is common that the alphabet depends on the dataset used, so
            it is recommended to pass it as an argument. Default is None.
        max_sequence_length : int, optional
            The maximum length of the sequence. Default is infinity.
        seed : int, optional
            Seed for random number generators. If None, no seed is set.
        batch_size : int, optional
            Number of molecules to process in parallel. If None, the default batch size is used.
        parallelize : bool, optional
            Whether to parallelize the docking process. Default is False.
        num_workers : int, optional
            Number of worker processes to use for parallelization. If None, the number of available CPU cores is used.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.
        force_isolation: bool, optional
            Whether to force the isolation of the black box. Default is False.

        Returns
        -------
        problem : Problem
            A problem instance containing the black box, and an initial value x0.

        Raises
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

        f = GSK3BetaBlackBox(
            string_representation=string_representation,
            alphabet=alphabet,
            max_sequence_length=max_sequence_length,
            force_isolation=force_isolation,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

        # Initial example (from the TDC docs)
        x0_smiles = "CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1"
        x0_selfies = translate_smiles_to_selfies([x0_smiles])[0]

        if string_representation.upper() == "SMILES":
            x0 = np.array([list(x0_smiles)])
        else:
            x0 = np.array([list(sf.split_selfies(x0_selfies))])

        gsk3_beta_problem = TDCProblem(
            black_box=f,
            x0=x0,
        )

        return gsk3_beta_problem
