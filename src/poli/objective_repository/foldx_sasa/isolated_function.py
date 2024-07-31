from pathlib import Path
from typing import List, Union

import numpy as np

from poli.core.proteins.foldx_isolated_function import FoldxIsolatedFunction
from poli.core.registry import register_isolated_function
from poli.core.util.proteins.foldx import FoldxInterface
from poli.core.util.proteins.mutations import find_closest_wildtype_pdb_file_to_mutant


class FoldXSASAIsolatedLogic(FoldxIsolatedFunction):
    """
    A black box implementation for computing the solvent accessible surface area (SASA) score using FoldX.

    Parameters
    -----------
    wildtype_pdb_path : Union[Path, List[Path]]
        The path(s) to the wildtype PDB file(s). Default is None.
    alphabet : List[str], optional
        The alphabet of amino acids. Default is None.
    experiment_id : str, optional
        The ID of the experiment. Default is None.
    tmp_folder : Path, optional
        The path to the temporary folder. Default is None.
    eager_repair : bool, optional
        Whether to perform eager repair. Default is False.
    verbose : bool, optional
        Flag indicating whether we print the output from FoldX. Default is False.

    Notes
    -----
    We expect the user to have FoldX v5.0 installed and compiled.
    More specifically, we expect a binary file called foldx to be
    in the path ~/foldx/foldx.
    """

    def __init__(
        self,
        wildtype_pdb_path: Union[Path, List[Path]],
        experiment_id: str = None,
        tmp_folder: Path = None,
        eager_repair: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            wildtype_pdb_path=wildtype_pdb_path,
            experiment_id=experiment_id,
            tmp_folder=tmp_folder,
            eager_repair=eager_repair,
            verbose=verbose,
        )

    def __call__(self, x: np.ndarray, context: None) -> np.ndarray:
        """Computes the SASA score for a given mutation x.

        Runs the given input x and pdb files provided
        in the context through FoldX and returns the
        total SASA score.

        Parameters
        -----------
        x : np.ndarray
            The input array representing the mutations.
        context : None
            The context for the black box computation.

        Returns
        --------
        np.ndarray
            The computed SASA score(s) as a numpy array.
        """
        # TODO: add support for multiple mutations.
        # For now, we assume that the batch size is
        # always 1.
        assert x.shape[0] == 1, "We only support single mutations for now. "

        # Create a working directory for this function call
        working_dir = self.create_working_directory()

        # Given x, we simply define the
        # mutations to be made as a mutation_list.txt
        # file.
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

        foldx_interface = FoldxInterface(working_dir, self.verbose)
        sasa_score = foldx_interface.compute_sasa(
            pdb_file=wildtype_pdb_file,
            mutations=mutations_as_strings,
        )

        return np.array([sasa_score]).reshape(-1, 1)


if __name__ == "__main__":
    register_isolated_function(
        FoldXSASAIsolatedLogic,
        name="foldx_sasa__isolated",
        conda_environment_name="poli__protein",
    )
