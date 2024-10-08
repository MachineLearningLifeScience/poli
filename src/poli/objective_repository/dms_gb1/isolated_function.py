"""
This module implements GB1 DMS,
using the open-available data from [1].

The black-box is a lookup of the combinatorically complete mutation landscape.
If the proposed sequence exists the associated value is returned, else a nan/inf value is returned.

The dataset reference is [2].

[1] "Active Learning-Assisted Directed Evolution"
Jason Yang, Ravi G. Lal, James C. Bowden, Raul Astudillo, Mikhail A. Hameedi, Sukhvinder Kaur, Matthew Hill, Yisong Yue, Frances H. Arnold
bioRxiv 2024.07.27.605457; doi: https://doi.org/10.1101/2024.07.27.605457.
[2] "Learning protein fitness landscapes with deep mutational scanning data from multiple sources" 
Lin Chen, Zehong Zhang, Zhenghao Li, Rui Li, Ruifeng Huo, Lifan Chen, Dingyan Wang, Xiaomin Luo, Kaixian Chen, Cangsong Liao, Mingyue Zheng,
Cell Systems,
Volume 14, Issue 8, 2023,
ISSN 2405-4712; doi: https://doi.org/10.1016/j.cels.2023.07.003.

"""

from __future__ import annotations

from pathlib import Path
from time import time
from typing import List, Union
from uuid import uuid4

import numpy as np
import pandas as pd

from poli.core.abstract_isolated_function import AbstractIsolatedFunction

THIS_DIR = Path(__file__).parent.resolve()


class DMSGB1IsolatedLogic(AbstractIsolatedFunction):
    """
    GB1 internal implementation.

    Parameters
    ----------
    penalize_unfeasible_with: float, optional
        The value to return when the input is unfeasible, by default None, which means that we raise an error when
        an unfeasible sequence (e.g. one with a length different
        from the wildtypes) is passed.
    alphabet : List[str], optional
        The alphabet for the problem, by default we use
        the amino acid list provided in poli.core.util.proteins.defaults.
    experiment_id : str, optional
        The experiment ID, by default None.

    Methods
    -------
    _black_box(x, context=None)
        The main black box method that performs the computation, i.e.
        it computes the stability of the mutant(s) in x.
    _load_dms_data()
        This function loads the DMS data under assets.

    Notes
    -----
    - If experiment_id is not provided, it is generated using the current timestamp and a random UUID.
    """

    def __init__(
        self,
        penalize_unfeasible_with: float | None = np.inf,
        experiment_id: str = None,
    ):
        """
        Initialize the GB1 Register object.

        Parameters:
        -----------
        penalize_unfeasible_with: float, optional
            The value to return when the input is unfeasible, by default None, which means that we raise an error when
            an unfeasible sequence (e.g. one with a length different
            from the wildtypes) is passed.
        experiment_id : str, optional
            The experiment ID, by default None.
        tmp_folder : Path, optional
            The temporary folder path, by default None.

        Notes:
        ------
        - If experiment_id is not provided, it is generated using the current timestamp and a random UUID.
        """
        assert wildtype_pdb_path is not None, (
            "Missing required argument wildtype_pdb_file. "
            "Did you forget to pass it to create and into the black box?"
        )

        if isinstance(wildtype_pdb_path, str):
            wildtype_pdb_path = Path(wildtype_pdb_path.strip())

        if isinstance(wildtype_pdb_path, Path):
            wildtype_pdb_path = [wildtype_pdb_path]

        if isinstance(wildtype_pdb_path, list):
            if isinstance(wildtype_pdb_path[0], str):
                # Assuming that wildtype_pdb_path is a list of strings
                wildtype_pdb_path = [Path(x.strip()) for x in wildtype_pdb_path]
            elif isinstance(wildtype_pdb_path[0], Path):
                pass

        # By this point, we can make sure the wildtype_pdb_path
        # is a list of Path objects.
        assert all([x.exists() for x in wildtype_pdb_path]), (
            "One of the wildtype PDBs does not exist. "
            "Please check the path and try again."
        )

        self.wildtype_pdb_paths = wildtype_pdb_path

        self.wildtype_residue_strings = [
            "".join(amino_acids) for amino_acids in self.wildtype_amino_acids
        ]

        # Validating the chains to keep
        if isinstance(chains_to_keep, type(None)):
            # Defaulting to always keeping chain A.
            chains_to_keep = ["A"] * len(self.wildtype_pdb_paths)

        if isinstance(chains_to_keep, str):
            chains_to_keep = [chains_to_keep] * len(self.wildtype_pdb_paths)

        if isinstance(chains_to_keep, list):
            assert len(chains_to_keep) == len(self.wildtype_pdb_paths), (
                "The number of chains to keep must be the same as the number of wildtypes."
                " You can specify a single chain to keep for all wildtypes, or a list of chains."
            )

        self.penalize_unfeasible_with = penalize_unfeasible_with

        if experiment_id is None:
            experiment_id = f"{int(time())}_{str(uuid4())[:8]}"
        self.experiment_id = experiment_id

        x0_pre_array = []
        for clean_wildtype_pdb_file in self.clean_wildtype_pdb_files:
            # Loads up the wildtype pdb files as strings
            wildtype_string = self.parse_pdb_as_residue_strings(clean_wildtype_pdb_file)
            x0_pre_array.append(list(wildtype_string))

        # Padding all of them to the longest sequence
        max_len = max([len(x) for x in x0_pre_array])
        x0_pre_array = [x + [""] * (max_len - len(x)) for x in x0_pre_array]

        self.x0 = np.array(x0_pre_array)

    def _clean_wildtype_pdb_files(self):
        """
        This function cleans the wildtype pdb files
        stored in self.wildtype_pdb_paths, using
        cached results if they exist.
        """

        # We make sure the PDBs are cleaned.
        # TODO: add chain to the name of the file
        for wildtype_pdb_path, chain_to_keep in zip(
            self.wildtype_pdb_paths, self.chains_to_keep
        ):
            if not (
                self.working_dir
                / "raw"
                / f"{wildtype_pdb_path.stem}_{chain_to_keep}.pdb"
            ).exists():
                self.rasp_interface.raw_pdb_to_unique_chain(
                    wildtype_pdb_path, chain=chain_to_keep
                )

            if not (
                self.working_dir
                / "cleaned"
                / f"{wildtype_pdb_path.stem}_{chain_to_keep}_clean.pdb"
            ).exists():
                self.rasp_interface.unique_chain_to_clean_pdb(wildtype_pdb_path)

            if not (
                self.working_dir
                / "parsed"
                / f"{wildtype_pdb_path.stem}_{chain_to_keep}_clean_coordinate_features.npz"
            ).exists():
                self.rasp_interface.cleaned_to_parsed_pdb(wildtype_pdb_path)

        self.clean_wildtype_pdb_files = [
            self.working_dir
            / "cleaned"
            / f"{wildtype_pdb_path.stem}_query_protein_uniquechain_clean.pdb"
            for wildtype_pdb_path in self.wildtype_pdb_paths
        ]

    def _load_dms_data(self) -> pd.DataFrame:
        pd.read_csv(THIS_DIR / "assets" / "fitnes.csv")

    def __call__(self, x, context=None):
        """
        Computes the stability of the mutant(s) in x.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape [b, L] containing strings.
        context : dict, optional
            Additional context information (default is None).

        Returns
        -------
        y : np.ndarray
            The stability of the mutant(s) in x.

        Notes
        -----
        - x is a np.array[str] of shape [b, L], where L is the length
          of the longest sequence in the batch, and b is the batch size.
          We process it by concantenating the array into a single string,
          where we assume the padding to be an empty string (if there was any).
          Each of these x_i's will be matched to the wildtype in
          self.wildtype_residue_strings with the lowest Hamming distance.
        """
        # We need to find the closest wildtype to each of the
        # sequences in x. For this, we need to compute the
        # Hamming distance between each of the sequences in x
        # and each of the wildtypes in self.wildtype_residue_strings.

        # closest_wildtypes will be a dictionary
        # of the form {wildtype_path: List[str] of mutations}
        # closest_wildtypes = defaultdict(list)
        # mutant_residue_strings = []
        # mutant_residue_to_hamming_distances = dict()
        results = []
        for x_i in x:
            # Assuming x_i is an array of strings
            mutant_residue_string = "".join(x_i)
            result = self._compute_mutant_residue_string_ddg(mutant_residue_string)
            results.append(result)

        return -np.array(results).reshape(-1, 1)


if __name__ == "__main__":
    from poli.core.registry import register_isolated_function

    register_isolated_function(
        RaspIsolatedLogic,
        name="rasp__isolated",
        conda_environment_name="poli__rasp",
        force=True,
    )
