"""
This script registers the stability and SASA FoldX black box
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
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from Bio.SeqUtils import seq1

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.proteins.pdb_parsing import parse_pdb_as_residues
from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli.core.util.proteins.mutations import (
    find_closest_wildtype_pdb_file_to_mutant,
)
from poli.core.util.proteins.foldx import FoldxInterface

from poli.core.proteins.foldx_black_box import FoldxBlackBox

from poli.core.util.seeding import seed_numpy, seed_python


class FoldXStabilityAndSASABlackBox(FoldxBlackBox):
    """
    A black box implementation for computing the solvent accessible surface area (SASA) score using FoldX.

    Parameters:
    -----------
    info : ProblemSetupInformation, optional
        The problem setup information. Default is None.
    batch_size : int, optional
        The batch size for parallel processing. Default is None.
    parallelize : bool, optional
        Whether to parallelize the computation. Default is False.
    num_workers : int, optional
        The number of workers for parallel processing. Default is None.
    evaluation_budget:  int, optional
        The maximum number of function evaluations. Default is infinity.
    wildtype_pdb_path : Union[Path, List[Path]], required
        The path(s) to the wildtype PDB file(s). Default is None.
    alphabet : List[str], optional
        The alphabet of amino acids. Default is None.
    experiment_id : str, optional
        The ID of the experiment. Default is None.
    tmp_folder : Path, optional
        The path to the temporary folder. Default is None.
    eager_repair : bool, optional
        Whether to perform eager repair. Default is False.
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
        we return the negative of the total energy.

        Parameters:
        -----------
        x : np.ndarray
            The input array representing the mutations.
        context : None
            The context for the black box computation.

        Returns:
        --------
        y: np.ndarray
            The computed stability and SASA score(s) as a numpy array.
        """
        # TODO: add support for multiple mutations.
        # For now, we assume that the batch size is
        # always 1.
        assert x.shape[0] == 1, "We only support single mutations for now. "

        # We create a different folder for each
        # mutation. This is because FoldX will
        # create a bunch of files in the working
        # directory, and we don't want to overwrite
        # them.
        working_dir = self.create_working_directory()

        # We only need to provide the mutations as
        # amino acid sequences. The FoldxInterface
        # will take care of the rest.
        mutations_as_strings = [
            "".join([amino_acid for amino_acid in x_i]) for x_i in x
        ]

        # We find the associated wildtype to this given
        # mutation. This is done by minimizing the
        # Hamming distance between the mutated residue
        # string and the wildtype residue strings of
        # all the PDBs we have.
        # TODO: this assumes a batch size of 1.
        wildtype_pdb_file = find_closest_wildtype_pdb_file_to_mutant(
            self.wildtype_pdb_paths, mutations_as_strings[0]
        )

        foldx_interface = FoldxInterface(working_dir)
        stability, sasa_score = foldx_interface.compute_stability_and_sasa(
            pdb_file=wildtype_pdb_file,
            mutations=mutations_as_strings,
        )

        return np.array([stability, sasa_score]).reshape(-1, 2)


class FoldXStabilityAndSASAProblemFactory(AbstractProblemFactory):
    """
    Factory class for creating FoldX SASA (Solvent Accessible Surface Area) problems.

    Methods:
    --------
    get_setup_information:
        Returns the setup information for the problem.
    create:
        Creates a problem instance with the specified parameters.
    """

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
            name="foldx_stability_and_sasa",
            max_sequence_length=np.inf,
            alphabet=alphabet,
            aligned=False,
        )

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        wildtype_pdb_path: Union[Path, List[Path]] = None,
        alphabet: List[str] = None,
        experiment_id: str = None,
        tmp_folder: Path = None,
        eager_repair: bool = False,
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        """
        Create a FoldXSASABlackBox object and compute the initial values of wildtypes.

        Parameters:
        ----------
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
        wildtype_pdb_path : Union[Path, List[Path]], required
            Path or list of paths to the wildtype PDB files.
        alphabet : List[str], optional
            List of amino acid symbols.
        experiment_id : str, optional
            Identifier for the experiment.
        tmp_folder : Path, optional
            Path to the temporary folder for intermediate files.
        eager_repair : bool, optional
            Flag indicating whether to perform eager repair.

        Returns:
        -------
        Tuple[AbstractBlackBox, np.ndarray, np.ndarray]
            A tuple containing the FoldXSASABlackBox object, the initial wildtype sequences, and the initial fitness values.

        Raises:
        ------
        ValueError
            If wildtype_pdb_path is missing or has an invalid type.
        """
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

        # We use the default alphabet if None was provided.
        # See ENCODING in foldx_utils.py
        if alphabet is None:
            alphabet = self.get_setup_information().get_alphabet()

        problem_info = self.get_setup_information()
        f = FoldXStabilityAndSASABlackBox(
            info=problem_info,
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

        # We need to compute the initial values of all wildtypes
        # in wildtype_pdb_path. For this, we need to specify x0,
        # a vector of wildtype sequences. These are padded to
        # match the maximum length with empty strings.
        wildtype_amino_acids_ = []
        for pdb_file in wildtype_pdb_path:
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

    foldx_problem_factory = FoldXStabilityAndSASAProblemFactory()
    register_problem(
        foldx_problem_factory,
        conda_environment_name="poli__protein",
    )
