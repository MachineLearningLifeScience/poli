"""
This module implements Rapid Stability Predictions (RaSP),
using their open-source code [1].

From a bird's eye view, RaSP is a black box that starts
with a collection of wildtype pdb files, and assesses the
stability of a (single-site) mutant. For the informed reader,
this black box can be considered a drop-in replacement of FoldX,
or Rosetta.

[1] TODO: add reference and implementation.
"""
from typing import Union, List
from pathlib import Path
from uuid import uuid4
from time import time

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.proteins.rasp import (
    RaspInterface,
    load_cavity_and_downstream_models,
)

from poli.core.util.proteins.pdb_parsing import (
    parse_pdb_as_residue_strings,
    parse_pdb_as_residues,
)
from poli.core.util.proteins.mutations import find_closest_wildtype_pdb_file_to_mutant

RASP_NUM_ENSEMBLE = 10
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


class RaspBlackBox(AbstractBlackBox):
    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        wildtype_pdb_path: Union[Path, List[Path]] = None,
        chains_to_keep: List[str] = None,
        alphabet: List[str] = None,
        experiment_id: str = None,
        tmp_folder: Path = None,
    ):
        """
        TODO: document
        """
        assert wildtype_pdb_path is not None, (
            "Missing required argument wildtype_pdb_file. "
            "Did you forget to pass it to create and into the black box?"
        )

        super().__init__(
            info=info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
        )

        if alphabet is None:
            alphabet = info.alphabet

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

    def _black_box(self, x, context=None):
        """
        TODO: document

        - x is a np.array[str] of shape [b, L], where L is the length
          of the longest sequence in the batch, and b is the batch size.
          We process it by concantenating the array into a single string,
          where we assume the padding to be an empty string (if there was any).
          Each of these x_i's will be matched to the wildtype in self.  wildtype_residue_strings with the lowest Hamming distance.
        """
        # Creating an interface for this experiment id

        # We need to find the closest wildtype to each of the
        # sequences in x. For this, we need to compute the
        # Hamming distance between each of the sequences in x
        # and each of the wildtypes in self.wildtype_residue_strings.
        closest_wildtypes = []
        mutant_residue_strings = []
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

            if hamming_distance > 1:
                raise ValueError("RaSP is only able to simulate single mutations.")

            closest_wildtypes.append(closest_wildtype_pdb_file)
            mutant_residue_strings.append(mutant_residue_string)

        # STEP 2:
        # Creating the dataframe with the relevant mutations
        df_structure = self.rasp_interface.create_df_structure(
            closest_wildtype_pdb_file,
            mutant_residue_strings=mutant_residue_strings,
        )

        # STEP 3:
        # Predicting
        cavity_model_net, ds_model_net = load_cavity_and_downstream_models()
        dataset_key = "predictions"
        df_ml = self.rasp_interface.predict(
            cavity_model_net,
            ds_model_net,
            df_structure,
            dataset_key,
            RASP_NUM_ENSEMBLE,
            RASP_DEVICE,
        )

        return df_ml["score_ml"].values.reshape(x.shape[0], 1)


if __name__ == "__main__":
    THIS_DIR = Path(__file__).parent.resolve()
    wildtype_pdb_path = THIS_DIR / "101m.pdb"
    chain_to_keep = "A"

    cavity_model_net, ds_model_net = load_cavity_and_downstream_models()
    rasp_interface = RaspInterface(THIS_DIR / "tmp")

    rasp_interface.raw_pdb_to_unique_chain(wildtype_pdb_path, chain_to_keep)
    rasp_interface.unique_chain_to_clean_pdb(wildtype_pdb_path)
    rasp_interface.cleaned_to_parsed_pdb(wildtype_pdb_path)

    df_structure = rasp_interface.create_df_structure(wildtype_pdb_path)
    print(df_structure.head())

    # Loading the models

    # Predicting
    dataset_key = "predictions"
    df_ml = rasp_interface.predict(
        cavity_model_net,
        ds_model_net,
        df_structure,
        dataset_key,
        RASP_NUM_ENSEMBLE,
        RASP_DEVICE,
    )

    print(df_ml.head())
