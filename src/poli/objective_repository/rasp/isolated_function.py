"""
This module implements Rapid Stability Predictions (RaSP),
using their open-source code [1].

From a bird's eye view, RaSP is a black box that starts
with a collection of wildtype pdb files, and assesses the
stability of a (single-site) mutant. For the informed reader,
this black box can be considered a drop-in replacement of FoldX,
or Rosetta.

[1] “Rapid Protein Stability Prediction Using Deep Learning Representations.”
Blaabjerg, Lasse M, Maher M Kassem, Lydia L Good, Nicolas Jonsson,
Matteo Cagiada, Kristoffer E Johansson, Wouter Boomsma, Amelie Stein,
and Kresten Lindorff-Larsen.  Edited by José D Faraldo-Gómez,
Detlef Weigel, Nir Ben-Tal, and Julian Echave. eLife 12
(May 2023): e82593. https://doi.org/10.7554/eLife.82593.

"""

from collections import defaultdict
from pathlib import Path
from time import time
from typing import List, Union
from uuid import uuid4

import numpy as np
import torch

from poli.core.abstract_isolated_function import AbstractIsolatedFunction
from poli.core.util.proteins.mutations import find_closest_wildtype_pdb_file_to_mutant
from poli.core.util.proteins.pdb_parsing import (
    parse_pdb_as_residue_strings,
    parse_pdb_as_residues,
)
from poli.core.util.proteins.rasp import (
    RaspInterface,
    load_cavity_and_downstream_models,
)

RASP_NUM_ENSEMBLE = 10
if torch.cuda.is_available():
    RASP_DEVICE = "cuda"
else:
    RASP_DEVICE = "cpu"

THIS_DIR = Path(__file__).parent.resolve()
HOME_DIR = THIS_DIR.home()
RASP_DIR = HOME_DIR / ".poli_objectives" / "rasp"
RASP_DIR.mkdir(parents=True, exist_ok=True)

# This is the folder where all the files
# generated by RaSP will be stored.
# Feel free to change it if you want
# to keep the files somewhere else by
# passing tmp_folder to the black box.
# TODO: what happens if the user is on Windows?

# As a brief summary, this is what RaSP will
# do inside this folder: it will create 4
# subfolders inside it, one for each step
# of the pipeline (raw, cleaned, parsed, output).
#
#  - raw contains a version of the PDBs with only
#    the chain of interest.
#
#  - cleaned contains a version of the PDBs with
#    only the atoms of interest.
#
#  - parsed contains an .npz file with the variables
#    of interest, and
#
#  - output contains the predictions of the model.
DEFAULT_TMP_PATH = Path("/tmp").resolve()


class RaspIsolatedLogic(AbstractIsolatedFunction):
    """
    RaSP internal implementation.

    Parameters
    ----------
    wildtype_pdb_path : Union[Path, List[Path]]
        The path(s) to the wildtype PDB file(s), by default None.
    additive : bool, optional
        Whether we treat multiple mutations as additive, by default False.
        If you are interested in running this black box with multiple
        mutations, you should set this to True. Otherwise, it will
        raise an error if you pass a sequence with more than one mutation.
    chains_to_keep : List[str], optional
        The chains to keep in the PDB file(s), by default we
        keep the chain "A" for all pdbs passed.
    alphabet : List[str], optional
        The alphabet for the problem, by default we use
        the amino acid list provided in poli.core.util.proteins.defaults.
    experiment_id : str, optional
        The experiment ID, by default None.
    tmp_folder : Path, optional
        The temporary folder path, by default None.

    Methods
    -------
    _black_box(x, context=None)
        The main black box method that performs the computation, i.e.
        it computes the stability of the mutant(s) in x.
    _clean_wildtype_pdb_files()
        This function cleans the wildtype pdb files
        stored in self.wildtype_pdb_paths, using
        cached results if they exist.


    Raises
    ------
    AssertionError
        If wildtype_pdb_path is not provided.

    Notes
    -----
    - The wildtype_pdb_path can be a single Path object or a list of Path objects.
    - If chains_to_keep is not provided, it defaults to keeping chain A for all wildtypes.
    - If experiment_id is not provided, it is generated using the current timestamp and a random UUID.
    - If tmp_folder is not provided, it defaults to the default temporary path.
    """

    def __init__(
        self,
        wildtype_pdb_path: Union[Path, List[Path]],
        additive: bool = False,
        chains_to_keep: List[str] = None,
        experiment_id: str = None,
        tmp_folder: Path = None,
    ):
        """
        Initialize the RaSP Register object.

        Parameters:
        -----------
        wildtype_pdb_path : Union[Path, List[Path]]
            The path(s) to the wildtype PDB file(s).
        additive : bool, optional
            Whether we treat multiple mutations as additive, by default False.
            If you are interested in running this black box with multiple
            mutations, you should set this to True. Otherwise, it will
            raise an error if you pass a sequence with more than one mutation.
        chains_to_keep : List[str], optional
            The chains to keep in the PDB file(s), by default we
            keep the chain "A" for all pdbs passed.
        alphabet : List[str], optional
            The alphabet for the problem, by default we use
            the amino acid list provided in poli.core.util.proteins.defaults.
        experiment_id : str, optional
            The experiment ID, by default None.
        tmp_folder : Path, optional
            The temporary folder path, by default None.

        Raises:
        -------
        AssertionError
            If wildtype_pdb_path is not provided.

        Notes:
        ------
        - The wildtype_pdb_path can be a single Path object or a list of Path objects.
        - If chains_to_keep is not provided, it defaults to keeping chain A for all wildtypes.
        - If experiment_id is not provided, it is generated using the current timestamp and a random UUID.
        - If tmp_folder is not provided, it defaults to the default temporary path.
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

        self.wildtype_resiudes = [
            parse_pdb_as_residues(pdb_file) for pdb_file in wildtype_pdb_path
        ]

        self.wildtype_amino_acids = [
            parse_pdb_as_residue_strings(pdb_file) for pdb_file in wildtype_pdb_path
        ]

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

        # At this point, we are sure that chains_to_keep is a list of strings
        self.chains_to_keep = chains_to_keep

        if experiment_id is None:
            experiment_id = f"{int(time())}_{str(uuid4())[:8]}"
        self.experiment_id = experiment_id

        self.tmp_folder = tmp_folder if tmp_folder is not None else DEFAULT_TMP_PATH

        # We need to preprocess these pdbs if they haven't
        # been preprocessed. This calls for a new folder
        # in tmp where _all_ pdbs are preprocessed and
        # stored... (?)
        self.clean_wildtype_pdb_files = None

        # After this, self.clean_wildtype_pdb_files is a
        # list of Path objects.
        self.working_dir = self.tmp_folder / "RaSP_tmp_files" / self.experiment_id
        self.rasp_interface = RaspInterface(working_dir=self.working_dir)
        self._clean_wildtype_pdb_files()

        x0_pre_array = []
        for clean_wildtype_pdb_file in self.clean_wildtype_pdb_files:
            # Loads up the wildtype pdb files as strings
            wildtype_string = self.parse_pdb_as_residue_strings(clean_wildtype_pdb_file)
            x0_pre_array.append(list(wildtype_string))

        # Padding all of them to the longest sequence
        max_len = max([len(x) for x in x0_pre_array])
        x0_pre_array = [x + [""] * (max_len - len(x)) for x in x0_pre_array]

        self.x0 = np.array(x0_pre_array)
        self.additive = additive

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

    def parse_pdb_as_residue_strings(self, pdb_file: Path) -> List[str]:
        return parse_pdb_as_residue_strings(pdb_file)

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
        closest_wildtypes = defaultdict(list)
        mutant_residue_strings = []
        mutant_residue_to_hamming_distances = dict()
        for x_i in x:
            # Assuming x_i is an array of strings
            mutant_residue_string = "".join(x_i)
            (
                closest_wildtype_pdb_file,
                hamming_distance,
            ) = find_closest_wildtype_pdb_file_to_mutant(
                self.clean_wildtype_pdb_files,
                mutant_residue_string,
                return_hamming_distance=True,
            )

            if hamming_distance > 1 and not self.additive:
                raise ValueError(
                    "RaSP is only able to simulate single mutations."
                    " If you want to simulate multiple mutations,"
                    " you should set additive=True in the create method"
                    " or in the black box of RaSP."
                )

            closest_wildtypes[closest_wildtype_pdb_file].append(mutant_residue_string)
            mutant_residue_strings.append(mutant_residue_string)
            mutant_residue_to_hamming_distances[mutant_residue_string] = (
                hamming_distance
            )

        # Loading the models in preparation for inference
        cavity_model_net, ds_model_net = load_cavity_and_downstream_models()
        dataset_key = "predictions"

        # STEP 2 and 3:
        # Creating the dataframe with the relevant mutations
        # PER wildtype pdb file.

        # We will store the results in a dictionary
        # of the form {mutant_string: score}.
        results = {}
        for (
            closest_wildtype_pdb_file,
            mutant_residue_strings_for_wildtype,
        ) in closest_wildtypes.items():
            df_structure = self.rasp_interface.create_df_structure(
                closest_wildtype_pdb_file,
                mutant_residue_strings=mutant_residue_strings_for_wildtype,
            )

            # STEP 3:
            # Predicting
            df_ml = self.rasp_interface.predict(
                cavity_model_net,
                ds_model_net,
                df_structure,
                dataset_key,
                RASP_NUM_ENSEMBLE,
                RASP_DEVICE,
            )

            for (
                mutant_residue_string_for_wildtype
            ) in mutant_residue_strings_for_wildtype:
                sliced_values_for_mutant = df_ml["score_ml"][
                    df_ml["mutant_residue_string"] == mutant_residue_string_for_wildtype
                ].values
                results[mutant_residue_string_for_wildtype] = sliced_values_for_mutant

                if self.additive:
                    assert (
                        sliced_values_for_mutant.shape[0]
                        == mutant_residue_to_hamming_distances[
                            mutant_residue_string_for_wildtype
                        ]
                    ), (
                        " The number of predictions made for this mutant"
                        " is not equal to the Hamming distance between the"
                        " mutant and the wildtype.\n"
                        "This is an internal error in `poli`. Please report"
                        " this issue by referencing the RaSP problem.\n"
                        " https://github.com/MachineLearningLifeScience/poli/issues"
                    )

                    # If we are treating the mutations as additive
                    # the sliced values for mutant will be an array
                    # of length equal to the Hamming distance between
                    # the mutant and the wildtype. These are the individual
                    # mutation predictions made by RaSP. We need to sum
                    # them up to get the final score.
                    results[mutant_residue_string_for_wildtype] = np.sum(
                        sliced_values_for_mutant
                    )

        # To reconstruct the final score, we rely
        # on mutant_residue_strings, which is a list
        # of strings IN THE SAME ORDER as the input
        # vector x.
        return np.array(
            [
                results[mutant_residue_string]
                for mutant_residue_string in mutant_residue_strings
            ]
        ).reshape(-1, 1)


if __name__ == "__main__":
    from poli.core.registry import register_isolated_function

    register_isolated_function(
        RaspIsolatedLogic,
        name="rasp__isolated",
        conda_environment_name="poli__rasp",
        force=True,
    )
