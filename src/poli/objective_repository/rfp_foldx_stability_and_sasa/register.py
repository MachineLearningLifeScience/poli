"""
Registers the stability and SASA FoldX black box
and objective factory.

FoldX [1] is a simulator that allows for computing the difference
in free energy between a wildtype protein and a mutated protein.
We pair this with biopython [2] to compute the SASA of the mutated
protein.

[1] Schymkowitz, J., Borg, J., Stricher, F., Nys, R., Rousseau, F.,
    & Serrano, L. (2005). The FoldX web server: an online force field.
    Nucleic acids research, 33(suppl_2), W382-W388.
[2] Cock PA, Antao T, Chang JT, Chapman BA, Cox CJ, Dalke A, Friedberg I,
    Hamelryck T, Kauff F, Wilczynski B and de Hoon MJL (2009) Biopython:
    freely available Python tools for computational molecular biology and
    bioinformatics. Bioinformatics, 25, 1422-1423
"""

import warnings
from pathlib import Path
from typing import List, Union

import numpy as np

from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem import Problem
from poli.core.util.seeding import seed_numpy, seed_python
from poli.objective_repository.foldx_stability_and_sasa.register import (
    FoldXStabilityAndSASABlackBox,
)


class RFPFoldXStabilityAndSASAProblemFactory(AbstractProblemFactory):
    """
    Factory class for creating FoldX SASA (Solvent Accessible Surface Area) problems.

    Methods
    -------
    create:
        Creates a problem instance with the specified parameters.
    """

    def create(
        self,
        wildtype_pdb_path: Union[Path, List[Path]],
        n_starting_points: int = None,
        strict: bool = False,
        experiment_id: str = None,
        tmp_folder: Path = None,
        eager_repair: bool = False,
        verbose: bool = False,
        seed: int = None,
        batch_size: int = 1,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        """
        Create a RFPFoldXSASABlackBox object and compute the initial values of wildtypes.

        Parameters
        ----------
        wildtype_pdb_path : Union[Path, List[Path]]
            Path or list of paths to the wildtype PDB files.
        n_starting_points: int, optional
            Size of D_0. Default is all available data.
            The minimum number of sequence is given by the Pareto front of the RFP problem, ie. you cannot have less sequences than that.
        strict: bool, optional
            Enable RuntimeErrors if number of starting sequences different to requested number of sequences.
        experiment_id : str, optional
            Identifier for the experiment.
        tmp_folder : Path, optional
            Path to the temporary folder for intermediate files.
        eager_repair : bool, optional
            Flag indicating whether to perform eager repair.
        verbose : bool, optional
            Flag indicating whether to print the output of FoldX.
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
            An instance of the RFP problem, containing the black box and the initial wildtype sequences.

        Raises
        ------
        ValueError
            If wildtype_pdb_path is missing or has an invalid type.
        """
        if seed is not None:
            seed_numpy(seed)
            seed_python(seed)

        if wildtype_pdb_path is None:
            raise ValueError(
                "Missing required argument wildtype_pdb_path. "
                "Did you forget to pass it to create()?"
            )

        if isinstance(wildtype_pdb_path, str):
            wildtype_pdb_path = [Path(wildtype_pdb_path.strip())]
        elif isinstance(wildtype_pdb_path, Path):
            wildtype_pdb_path = [wildtype_pdb_path]
        elif isinstance(wildtype_pdb_path, list):
            if isinstance(wildtype_pdb_path[0], str):
                wildtype_pdb_path = [Path(x.strip()) for x in wildtype_pdb_path]
            elif isinstance(wildtype_pdb_path[0], Path):
                pass
        else:
            raise ValueError(
                f"wildtype_pdb_path must be a string or a Path. Received {type(wildtype_pdb_path)}"
            )
        # By this point, we know that wildtype_pdb_path is a
        # list of Path objects.
        if n_starting_points is None:
            n_starting_points = len(wildtype_pdb_path)

        # For a comparable RFP definition we require the sequences of the Pareto front:
        pareto_sequences_name_pdb_dict = {
            "DsRed.M1": "2VAD",
            "DsRed.T4": "2VAE",
            "mScarlet": "5LK4",
            "AdRed": "6AA7",
            "mRouge": "3NED",
            "RFP630": "3E5V",
        }

        if strict and n_starting_points < len(pareto_sequences_name_pdb_dict):
            raise RuntimeError(
                f"Initial number of sequences too low!\nMinimum size {len(pareto_sequences_name_pdb_dict)} , requested {n_starting_points}"
            )

        remaining_n_starting_points = max(
            n_starting_points - len(pareto_sequences_name_pdb_dict.values()), 0
        )
        # filter minimal required Pareto sequences
        pareto_pdb_files = [
            p
            for p in wildtype_pdb_path
            if any(
                [
                    bool(_pdb.lower() in str(p).lower())
                    for _pdb in pareto_sequences_name_pdb_dict.values()
                ]
            )
        ]
        if len(pareto_pdb_files) != len(pareto_sequences_name_pdb_dict):
            raise RuntimeError(
                f"The provided PDB files list is incomplete!\n Required={','.join(list(pareto_sequences_name_pdb_dict.values()))} provided files={pareto_pdb_files}"
            )

        remaining_wildtype_pdb_files = list(
            set(wildtype_pdb_path) - set(pareto_pdb_files)
        )
        np.random.shuffle(remaining_wildtype_pdb_files)
        remaining_wildtype_pdb_files = remaining_wildtype_pdb_files[
            :remaining_n_starting_points
        ]  # subselect w.r.t. requested number of sequences
        pdb_files_for_black_box: List = pareto_pdb_files + remaining_wildtype_pdb_files

        f = FoldXStabilityAndSASABlackBox(
            wildtype_pdb_path=pdb_files_for_black_box,
            experiment_id=experiment_id,
            tmp_folder=tmp_folder,
            eager_repair=eager_repair,
            verbose=verbose,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            force_isolation=force_isolation,
        )

        # We need to compute the initial values of all wildtypes
        # in wildtype_pdb_path. For this, we need to specify x0,
        # a vector of wildtype sequences. These are padded to
        # match the maximum length with empty strings.
        wildtype_amino_acids_ = f.wildtype_amino_acids
        longest_wildtype_length = max([len(x) for x in wildtype_amino_acids_])

        wildtype_amino_acids = [
            amino_acids + [""] * (longest_wildtype_length - len(amino_acids))
            for amino_acids in wildtype_amino_acids_
        ]

        x0 = np.array(wildtype_amino_acids).reshape(
            len(pdb_files_for_black_box), longest_wildtype_length
        )

        if n_starting_points is not None and x0.shape[0] != n_starting_points:
            if strict:
                raise RuntimeError(
                    f"Requested number of starting sequences different to loaded!\nRequested n={n_starting_points}, loaded n={x0.shape[0]}"
                )
            else:
                warnings.warn(
                    f"Requested number of starting sequences different to loaded!\nRequested n={n_starting_points}, loaded n={x0.shape[0]}"
                )

        problem = Problem(f, x0)

        return problem
