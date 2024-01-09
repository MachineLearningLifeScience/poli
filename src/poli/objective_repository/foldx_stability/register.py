"""
This script registers the stability FoldX black box and objective factory.

FoldX [1] is a simulator that allows for computing the difference
in free energy between a wildtype protein and a mutated protein. We 
also use biopython for pre-processing the PDB files [2].

[1] Schymkowitz, J., Borg, J., Stricher, F., Nys, R., Rousseau, F.,
    & Serrano, L. (2005). The FoldX web server: an online force field.
    Nucleic acids research, 33(suppl_2), W382-W388.
[2] Cock PA, Antao T, Chang JT, Chapman BA, Cox CJ, Dalke A, Friedberg I,
    Hamelryck T, Kauff F, Wilczynski B and de Hoon MJL (2009) Biopython:
    freely available Python tools for computational molecular biology and 
    bioinformatics. Bioinformatics, 25, 1422-1423
"""
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from Bio.SeqUtils import seq1

from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.proteins.pdb_parsing import (
    parse_pdb_as_residues,
)
from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli.core.util.proteins.mutations import (
    find_closest_wildtype_pdb_file_to_mutant,
)
from poli.core.util.proteins.foldx import FoldxInterface

from poli.core.proteins.foldx_black_box import FoldxBlackBox

from poli.core.util.seeding import seed_numpy, seed_python

# This is the folder where all the files
# generated by FoldX will be stored.
# Feel free to change it if you want
# to keep the files somewhere else.
# An alternative is e.g. TMP_PATH = THIS_DIR
# TODO: what happens if the user is on Windows?
# TMP_PATH = THIS_DIR / "tmp"
TMP_PATH = Path("/tmp").resolve()


class FoldXStabilityBlackBox(FoldxBlackBox):
    """
    A black box implementation for evaluating the stability of protein structures using FoldX.

    Parameters
    ----------
    info : ProblemSetupInformation
        The problem setup information (usually provided by the factory).
    wildtype_pdb_path : Union[Path, List[Path]]
        The path(s) to the wildtype PDB file(s).
    alphabet : List[str], optional
        The alphabet of amino acids. By default, we use the 20
        amino acids shown in poli.core.util.proteins.defaults.
    experiment_id : str, optional
        The ID of the experiment (default is None).
    tmp_folder : Path, optional
        The path to the temporary folder (default is None).
    eager_repair : bool, optional
        Whether to eagerly repair the protein structures (default is False).
    batch_size : int, optional
        The batch size for parallel processing (default is None).
    parallelize : bool, optional
        Whether to parallelize the computation (default is False).
    num_workers : int, optional
        The number of workers for parallel processing (default is None).
    evaluation_budget:  int, optional
        The maximum number of function evaluations. Default is infinity.

    Methods
    -------
    _black_box(x, context)
        Runs the given input x and pdb files provided in the context through FoldX and returns the total stability score.

    Notes
    -----
    We expect the user to have FoldX v5.0 installed and compiled.
    More specifically, we expect a binary file called foldx to be
    in the path ~/foldx/foldx.
    """

    def __init__(
        self,
        info: ProblemSetupInformation = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        wildtype_pdb_path: Union[Path, List[Path]] = None,
        alphabet: List[str] = None,
        experiment_id: str = None,
        tmp_folder: Path = None,
        eager_repair: bool = False,
    ):
        super().__init__(
            info=info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
            wildtype_pdb_path=wildtype_pdb_path,
            alphabet=alphabet,
            experiment_id=experiment_id,
            tmp_folder=tmp_folder,
            eager_repair=eager_repair,
        )

    def _black_box(self, x: np.ndarray, context: None) -> np.ndarray:
        """
        Runs the given input x and pdb files provided
        in the context through FoldX and returns the
        total energy score.

        Since the goal is MINIMIZING the energy,
        we return the negative of the total energy
        (a.k.a. the stability).

        Parameters
        ----------
        x : np.ndarray
            The input array representing the mutations.
        context : None
            The context for the computation.

        Returns
        -------
        np.ndarray
            The array of stability scores.

        """
        # TODO: add support for multiple mutations.
        # For now, we assume that the batch size is
        # always 1.
        assert x.shape[0] == 1, "We only support single mutations for now. "

        # self.wildtype_pdb_paths is a list of paths
        wildtype_pdb_paths = self.wildtype_pdb_paths

        # Create a working directory for this function call
        working_dir = self.create_working_directory()

        # Given that x, we simply define the
        # mutations to be made as a mutation_list.txt
        # file.
        mutations_as_strings = [
            "".join([amino_acid for amino_acid in x_i]) for x_i in x
        ]

        # We find the associated wildtype to this given
        # mutation. This is done by minimizing the
        # Hamming distance between the wildtype
        # residue strings of all the PDBs we have.
        # TODO: this assumes a batch size of 1.
        wildtype_pdb_path = find_closest_wildtype_pdb_file_to_mutant(
            wildtype_pdb_paths, mutations_as_strings[0]
        )

        foldx_interface = FoldxInterface(working_dir)
        stability = foldx_interface.compute_stability(
            pdb_file=wildtype_pdb_path, mutations=mutations_as_strings
        )

        return np.array([stability]).reshape(-1, 1)


class FoldXStabilityProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        """
        Get the setup information for the foldx_sasa objective.

        Returns
        -------
        ProblemSetupInformation
            The setup information for the objective.

        Notes
        -----
        By default, the method uses the 20 amino acids shown in
        poli.core.util.proteins.defaults.
        """
        alphabet = AMINO_ACIDS

        return ProblemSetupInformation(
            name="foldx_stability",
            max_sequence_length=np.inf,
            alphabet=alphabet,
            aligned=False,
        )

    def create(
        self,
        wildtype_pdb_path: Union[Path, List[Path]],
        alphabet: List[str] = None,
        experiment_id: str = None,
        tmp_folder: Path = None,
        eager_repair: bool = False,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ) -> Tuple[FoldXStabilityBlackBox, np.ndarray, np.ndarray]:
        """
        Creates a FoldX stability black box function and initial observations.

        Parameters
        ----------
        wildtype_pdb_path : Union[Path, List[Path]]
            Path(s) to the wildtype PDB file(s).
        alphabet : List[str], optional
            List of amino acids to use as the alphabet.
        experiment_id : str, optional
            Identifier for the experiment.
        tmp_folder : Path, optional
            Path to the temporary folder for storing intermediate files.
        eager_repair : bool, optional
            Whether to eagerly repair the protein structures.
        seed : int, optional
            Seed for random number generation.
        batch_size : int, optional
            Number of sequences to process in parallel.
        parallelize : bool, optional
            Whether to parallelize the computation.
        num_workers : int, optional
            Number of worker processes to use for parallel computation.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.

        Returns
        -------
        Tuple[AbstractBlackBox, np.ndarray, np.ndarray]
            A tuple containing the created black box function, the initial sequence(s), and the initial fitness value(s).

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

        if alphabet is None:
            # We use the default alphabet.
            # See AMINO_ACIDS in foldx_utils.py
            alphabet = self.get_setup_information().get_alphabet()

        problem_info = self.get_setup_information()
        # TODO: add support for a larger batch-size.
        f = FoldXStabilityBlackBox(
            info=problem_info,
            wildtype_pdb_path=wildtype_pdb_path,
            alphabet=alphabet,
            experiment_id=experiment_id,
            tmp_folder=tmp_folder,
            eager_repair=eager_repair,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )

        # During the creation of the black box,
        # we might have repaired the PDB files.
        # Thus, we need to compute the initial
        # values of all wildtypes in wildtype_pdb_path
        # using the repaired PDB files instead.
        repaired_wildtype_pdb_paths = f.wildtype_pdb_paths

        # We need to compute the initial values of all wildtypes
        # in wildtype_pdb_path. For this, we need to specify x0,
        # a vector of wildtype sequences. These are padded to
        # match the maximum length with empty strings.
        wildtype_amino_acids_ = []
        for pdb_file in repaired_wildtype_pdb_paths:
            wildtype_residues = parse_pdb_as_residues(pdb_file)
            wildtype_amino_acids_.append(
                [
                    seq1(residue.get_resname())
                    for residue in wildtype_residues
                    if residue.get_resname() != "NA"
                ]
            )
        longest_wildtype_length = max([len(x) for x in wildtype_amino_acids_])

        wildtype_amino_acids = [
            amino_acids + [""] * (longest_wildtype_length - len(amino_acids))
            for amino_acids in wildtype_amino_acids_
        ]

        x0 = np.array(wildtype_amino_acids).reshape(
            len(wildtype_pdb_path), longest_wildtype_length
        )

        f_0 = f(x0)

        return f, x0, f_0


if __name__ == "__main__":
    from poli.core.registry import register_problem

    foldx_problem_factory = FoldXStabilityProblemFactory()
    register_problem(
        foldx_problem_factory,
        conda_environment_name="poli__protein",
    )
