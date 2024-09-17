"""An interface to the original RaSP codebase.

This module takes and adapts RaSP's original implementation
(which can be found at [1]), and writes an interface that
handles the preprocessing and inference steps.

RaSP, like foldx [2], predicts the effect of mutations on protein
stability. However, it does so using a deep learning model
instead of actual simulations. The drawback being that
only additive mutations are supported (indeed, we currently
only support one-mutation-from-wildtype).

To perform inference, RaSP requires downloading the following:
- the `reduce` executable, which is used to clean the PDB files.
- the cavity model, which is used to predict the cavity features.
- the downstream models, which are used to predict the ddG values.

This means that you will need internet access to use this interface
(and the "rasp" objective function) for the first time. These models
and tools will be cached in the ~/.poli_objectives/rasp directory,
so you will only need internet access once.

RaSP's source code was provided with an Apache 2.0 license. Modifications
from the source have been duly noted. Most of the source code can be found
at ./inner_rasp.

References
----------
[1] “Rapid Protein Stability Prediction Using Deep Learning Representations.”
    Blaabjerg, Lasse M, Maher M Kassem, Lydia L Good, Nicolas Jonsson, Matteo Cagiada,
    Kristoffer E Johansson, Wouter Boomsma, Amelie Stein, and Kresten Lindorff-Larsen.
    Edited by José D Faraldo-Gómez, Detlef Weigel, Nir Ben-Tal, and Julian Echave.
    eLife 12 (May 2023): e82593. https://doi.org/10.7554/eLife.82593.
[2] The FoldX web server: an online force field.
    Schymkowitz, J., Borg, J., Stricher, F., Nys, R.,
    Rousseau, F., & Serrano, L. (2005).  Nucleic acids research,
    33(suppl_2), W382-W388.
"""

import logging
import os
import stat
import subprocess
import traceback
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from Bio.PDB.Polypeptide import index_to_one, one_to_index
from pdbtools.pdb_delhetatm import run as pdb_delhetatm_run
from pdbtools.pdb_delres import run as pdb_delres_run
from pdbtools.pdb_fixinsert import run as pdb_fixinsert_run
from pdbtools.pdb_selchain import run as pdb_selchain_run
from pdbtools.pdb_tidy import run as pdb_tidy_run

from poli.core.util.files.download_files_from_github import (
    download_file_from_github_repository,
)
from poli.core.util.files.integrity import compute_md5_from_filepath
from poli.core.util.proteins.mutations import edits_between_strings
from poli.core.util.proteins.rasp.inner_rasp.cavity_model import (
    ResidueEnvironmentsDataset,
)
from poli.core.util.proteins.rasp.inner_rasp.helpers import ds_pred
from poli.core.util.proteins.rasp.inner_rasp.pdb_parser_scripts.clean_pdb import (
    clean_pdb,
)
from poli.core.util.proteins.rasp.inner_rasp.pdb_parser_scripts.extract_environments import (
    extract_environments,
)

THIS_DIR = Path(__file__).parent.resolve()
HOME_DIR = THIS_DIR.home()
RASP_DIR = HOME_DIR / ".poli_objectives" / "rasp"
RASP_DIR.mkdir(parents=True, exist_ok=True)


class RaspInterface:
    def __init__(
        self,
        working_dir: Path,
        verbose: bool = False,
        verify_integrity_of_download: bool = True,
    ) -> None:
        self.working_dir = working_dir
        self.verbose = verbose
        self.verify_integrity_of_download = verify_integrity_of_download

        # Making the appropriate folders:
        (self.working_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.working_dir / "cleaned").mkdir(parents=True, exist_ok=True)
        (self.working_dir / "parsed").mkdir(parents=True, exist_ok=True)
        (self.working_dir / "output").mkdir(parents=True, exist_ok=True)

        # Downloading reduce and compile it.
        # TODO: should we be doing this eagerly? Or should
        # we wait to download and install it until we need it?
        self.reduce_executable_path = None
        self.get_and_compile_reduce()

        # At this point, we should have the reduce executable
        # at self.reduce_executable_path.

        # Downloading the cavity and downstream models.
        # TODO: should we be doing this eagerly? Or should
        # we wait to download and install it until we need it?
        self.download_cavity_and_downstream_models(
            verbose=verbose, verify_integrity_of_download=verify_integrity_of_download
        )
        self.alphabet = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]

    def get_and_compile_reduce(self):
        """
        This function downloads reduce and compiles it
        if we can't find it inside ~/.poli_objectives/rasp.
        """
        HOME_DIR = self.working_dir.home()
        RASP_DIR = HOME_DIR / ".poli_objectives" / "rasp"
        RASP_DIR.mkdir(parents=True, exist_ok=True)

        REDUCE_DIR = RASP_DIR / "reduce"
        EXECUTABLE_PATH = REDUCE_DIR / "reduce_src" / "reduce"
        if not (EXECUTABLE_PATH).exists():
            # Clone it using git.
            # TODO: Is there a way of downloading the contents
            # of the repo without cloning it?
            try:
                subprocess.run(
                    "git clone https://github.com/rlabduke/reduce.git",
                    cwd=RASP_DIR,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    check=True,
                )

                # Pinning to commit hash bd23a0bf...
                subprocess.run(
                    "git checkout bd23a0bf627ae9b08842102a5c2e9404b4a81924",
                    cwd=REDUCE_DIR,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    check=True,
                )
            except subprocess.CalledProcessError:
                # Printing the traceback
                traceback.print_exc()
                raise RuntimeError(
                    "Could not clone the reduce repository. "
                    "Please check your internet connection."
                )

            # compile it.
            try:
                subprocess.run(
                    "make",
                    cwd=REDUCE_DIR,
                    stdout=subprocess.DEVNULL,
                    check=True,
                )
            except subprocess.CalledProcessError:
                # TODO: should we be purging it ourselves?
                traceback.print_exc()
                raise RuntimeError(
                    "Something went wrong while compiling reduce. "
                    "Purge the folder ~/.poli_objectives/rasp/reduce "
                    " and try again."
                )

            # Change its permissions.
            os.chmod(EXECUTABLE_PATH, stat.S_IEXEC)

        self.reduce_executable_path = EXECUTABLE_PATH

    def predict(
        self,
        cavity_model_net,
        ds_model_net,
        df_structure,
        dataset_key,
        NUM_ENSEMBLE,
        DEVICE,
    ) -> pd.DataFrame:
        df_ml = ds_pred(
            cavity_model_net,
            ds_model_net,
            df_structure,
            dataset_key,
            NUM_ENSEMBLE,
            DEVICE,
        )

        df_total = df_structure.merge(
            df_ml, on=["pdbid", "chainid", "variant"], how="outer"
        )
        df_total = df_total.drop("resenv", axis=1)

        # Removed by MGD: we don't need the PRISM files, we only
        # need the dataframe.
        #
        # Parse output into separate files by pdb, print to PRISM format
        # for pdbid in df_total["pdbid"].unique():
        #     df_pdb = df_total[df_total["pdbid"] == pdbid]
        #     for chainid in df_pdb["chainid"].unique():
        #         pred_outdir = self.working_dir / "output" / f"{dataset_key}"
        #         pred_outdir.mkdir(parents=True, exist_ok=True)
        #         pred_outfile = pred_outdir / f"cavity_pred_{pdbid}_{chainid}.csv"
        #         print(
        #             f"Parsing predictions from pdb: {pdbid}{chainid} into {pred_outfile}"
        #         )
        #         df_chain = df_pdb[df_pdb["chainid"] == chainid]
        #         df_chain = df_chain.assign(pos=df_chain["variant"].str[1:-1])
        #         df_chain["pos"] = pd.to_numeric(df_chain["pos"])
        #         first_res_no = min(df_chain["pos"])
        #         df_chain = df_chain.assign(wt_AA=df_chain["variant"].str[0])
        #         df_chain = df_chain.assign(mt_AA=df_chain["variant"].str[-1])
        #         seq = get_seq_from_variant(df_chain)
        #         df_chain.to_csv(pred_outfile, index=False)
        #         prism_outfile = pred_outdir / f"prism_cavity_{pdbid}_{chainid}.txt"

        #         # if (AF_ID !=''):
        #         #   prism_outfile = f"/content/output/{dataset_key}/prism_cavity_{AF_ID}_{chainid}.txt"
        #         # elif (PDB_ID !=''):
        #         #   prism_outfile = f"/content/output/{dataset_key}/prism_cavity_{PDB_ID}_{chainid}.txt"
        #         # elif PDB_custom:
        #         #   prism_outfile = f"/content/output/{dataset_key}/prism_cavity_XXXX_{chainid}.txt"
        #         # cavity_to_prism(df_chain, pdbid, chainid, seq, prism_outfile)

        return df_total

    def download_cavity_and_downstream_models(
        self, verbose: bool = False, verify_integrity_of_download: bool = True
    ):
        """
        This function downloads the cavity and downstream models
        at the ~/.poli_objectives/rasp directory.

        If strict is True, then we will raise an error if the
        models we download don't match the expected MD5 checksums.
        Otherwise, we will just log a warning.
        """
        if os.environ.get("GITHUB_TOKEN_FOR_POLI") is None:
            logging.warning(
                "This black box objective function require downloading files "
                "from GitHub. Since the API rate limit is 60 requests per hour, "
                "we recommend creating a GitHub token and setting it as an "
                "environment variable called GITHUB_TOKEN_FOR_POLI. "
                "To create a GitHub token like this, follow the instructions here: "
                "https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token"
            )

        cavity_model_path = RASP_DIR / "cavity_model_15.pt"
        ds_models_paths = [
            RASP_DIR / "ds_models" / f"ds_model_{i}" / "model.pt" for i in range(10)
        ]

        repository_name = "KULL-Centre/papers"
        output_folder_in_repository = "2022/ML-ddG-Blaabjerg-et-al/output"
        commit_sha = "3ccebe87e017b6bd737f88e1943557d128c85616"

        # We will compare the md5 checksums of the models
        # we download against these pre-computed md5s.
        precomputed_model_md5_checksums = {
            cavity_model_path: "7f039dab359da67f5870e4c8ba4204ee",
            "ds_model_0": "88d8d836f4070245d23532ec46ff886b",
            "ds_model_1": "4ceca5dda221251f1333cc67717339f2",
            "ds_model_2": "9126cb60c340d2c165e05e7a8e5b3ffa",
            "ds_model_3": "d3cc21c451b1e5dbe797937fd6c34d3e",
            "ds_model_4": "a0d23110519a3763205a795fb713b5bf",
            "ds_model_5": "6c0bc930643aa51ebeba44ddea3b5865",
            "ds_model_6": "1ad01d6e52160f9f4083775f247d6127",
            "ds_model_7": "2470800d25b8c3c088c1eb17e6c2e440",
            "ds_model_8": "313b25729790e7698ce9f35b0b632177",
            "ds_model_9": "9f02a8de64a4a354eb4734d54ee9ba23",
        }

        if verbose:
            print("poli 🧪: Downloading the cavity and downstream models")
            print(f"Repository: {repository_name}")
            print(f"Commit: {commit_sha}")

        # Downloading the cavity model.
        if not cavity_model_path.exists():
            if verbose:
                print("poli 🧪: Downloading the cavity model")

            download_file_from_github_repository(
                repository_name=repository_name,
                file_path_in_repository=f"{output_folder_in_repository}/cavity_models/cavity_model_15.pt",
                download_path_for_file=cavity_model_path,
                commit_sha=commit_sha,
                exist_ok=True,
                verbose=verbose,
            )

            # Verifying the integrity of the file we just downloaded.
            downloaded_md5_checksum = compute_md5_from_filepath(cavity_model_path)
            if (
                downloaded_md5_checksum
                != precomputed_model_md5_checksums[cavity_model_path]
            ):
                if verify_integrity_of_download:
                    raise RuntimeError(
                        "The downloaded cavity model does not match the expected md5 checksum. "
                        "This could be due to a corrupted download, or a malicious attack. "
                        "Delete the files at ~/.poli_objectives/rasp and try again."
                    )
                else:
                    logging.warning(
                        "The downloaded cavity model does not match the expected md5 checksum. "
                        "This could be due to a corrupted download, or a malicious attack. "
                    )
        elif verbose:
            print("poli 🧪: Cavity model already exists. Skipping.")

        # Downloading the downstream models
        for path_ in ds_models_paths:
            if not path_.exists():
                if verbose:
                    print(f"poli 🧪: Downloading the downstream model {path_.parent}")

                local_path_in_directory = path_.relative_to(RASP_DIR)
                download_file_from_github_repository(
                    repository_name=repository_name,
                    file_path_in_repository=f"{output_folder_in_repository}/{local_path_in_directory}",
                    download_path_for_file=path_,
                    commit_sha=commit_sha,
                    exist_ok=False,
                )

                # Verifying the integrity of the file we just downloaded.
                downloaded_md5_checksum = compute_md5_from_filepath(path_)
                if (
                    downloaded_md5_checksum
                    != precomputed_model_md5_checksums[path_.parent.name]
                ):
                    if verify_integrity_of_download:
                        raise RuntimeError(
                            f"The downloaded downstream model {path_.parent.name} does not match the expected md5 checksum. "
                            "This could be due to a corrupted download, or a malicious attack. "
                            "Delete the files at ~/.poli_objectives/rasp and try again."
                        )
                    else:
                        logging.warning(
                            f"The downloaded downstream model {path_.name} does not match the expected md5 checksum. "
                            "This could be due to a corrupted download, or a malicious attack. "
                        )
            elif verbose:
                print(
                    f"poli 🧪: Downstream model {path_.parent.name} already exists. Skipping."
                )

    def raw_pdb_to_unique_chain(self, wildtype_pdb_path: Path, chain: str = "A"):
        """
        This function takes a raw pdb file, extracts a chain, and tidies up
        using pdbtools. Following their interactive implementation, this
        function will:
            1. Select a chain.
            2. Delete heteroatoms.
            3. Delete residues between -999 and 0.
            4. Fix insertions.
            5. Tidy the PDB.

        The output of this function is a PDB file with a single chain,
        stored at the working directory of this RaspInterface instance.
        More precisely, it will be stored at
        working_dir / raw / {wildtype_pdb_path.stem}_{chain}.pdb
        """
        # First command to run (which uses pdbtools to select a chain,
        # delete heteroatoms, remove some residues, fix insertions,
        # and tidy the PDB).
        raw_output_path = (
            self.working_dir
            / "raw"
            / f"{wildtype_pdb_path.stem}_query_protein_uniquechain.pdb"
        )

        # We need to load and transform the pdb file into
        # a line buffer that pdbtools can understand.
        with open(wildtype_pdb_path, "r") as fp:
            all_lines = fp.readlines()

        # Step 1: selecting the chain
        selchain_result = pdb_selchain_run(all_lines, chain)

        # Step 2: deleting heteroatoms
        deleting_heteroatoms_result = pdb_delhetatm_run(selchain_result)

        # Step 3: deleting residues between -999 and 0
        deleting_residues_result = pdb_delres_run(
            deleting_heteroatoms_result, residue_range=(-999, 0), step=1
        )

        # Step 4: fixing insertions
        fix_inserts_result = pdb_fixinsert_run(deleting_residues_result, [])

        # Step 5: tidying up the pdb
        tidy_result = pdb_tidy_run(fix_inserts_result)

        # tidy_result is now a generator of lines, we can use
        # it to write the output file.
        with open(raw_output_path, "w") as fp:
            for line in tidy_result:
                fp.write(line)

    def unique_chain_to_clean_pdb(self, wildtype_pdb_path: Path):
        """
        This function takes a pdb with a single chain, and
        cleans it using RaSP's clean_pdb.py script.

        The output of this function is a cleaned pdb file
        stored at the working directory of this RaspInterface
        instance. More precisely, it will be stored at
        working_dir / cleaned / {wildtype_pdb_path.stem}_clean.pdb
        This stem is usually just {pdb_id}_{chain}.
        """
        clean_pdb(
            pdb_input_filename=str(
                self.working_dir
                / "raw"
                / f"{wildtype_pdb_path.stem}_query_protein_uniquechain.pdb"
            ),
            out_dir=str(self.working_dir / "cleaned"),
            reduce_executable=str(self.reduce_executable_path),
        )

    def cleaned_to_parsed_pdb(
        self,
        wildtype_pdb_path: Path,
        max_radius: float = 9.0,
        include_center: bool = False,
    ):
        """
        This function takes a cleaned pdb file, and extracts the
        residue environments using extract_environments.py.
        The output of this function is a .npz file in the
        parsed directory given by:
        working_dir / parsed / {wildtype_pdb_path.stem}.npz
        At this stage, this stem is usually {pdb_id}_{chain}_clean
        """
        # The command used by the original RaSP implementation:
        # !python3 /content/src/pdb_parser_scripts/extract_environments.py --pdb_in /content/data/test/predictions/cleaned/query_protein_uniquechain_clean.pdb  --out_dir /content/data/test/predictions/parsed/  #&> /dev/null

        # We modify the naming convention to make it easier. Now
        # the output file will be named:
        pdb_id = wildtype_pdb_path.stem
        extract_environments(
            pdb_filename=str(
                self.working_dir
                / "cleaned"
                / f"{wildtype_pdb_path.stem}_query_protein_uniquechain_clean.pdb"
            ),
            pdb_id=pdb_id,
            max_radius=max_radius,
            out_dir=str(self.working_dir / "parsed"),
            include_center=include_center,
        )

    def create_df_structure(
        self, wildtype_pdb_path: Path, mutant_residue_strings: List[str] = None
    ):
        """
        This function creates a pandas dataframe with the
        residue environments of a given pdb file, and
        prepares it for ALL single-site mutations.

        If a list of mutant residue strings is provided,
        we will only consider the mutations in that list.
        For example, if the wildtype_pdb_path has the
        following residue string:

        "MSEMETKQV"

        and the mutant_residue_strings is:

        ["ASEMETKQV", "RSEMETKQV", "NSEMETKQV"]

        then the output dataframe will only contain the
        mutations M1A, M1R, and M1N.

        If no mutant_residue_strings is provided, then
        we will consider ALL single-site mutations.
        """
        # TODO: this is the convention inside the cleaning scripts,
        # but it feels flimsy. We should probably change it.
        pdb_id = wildtype_pdb_path.stem.replace("_query_protein_uniquechain_clean", "")
        pdb_filename_for_df = (
            self.working_dir / "parsed" / f"{pdb_id}_coordinate_features.npz"
        )
        assert pdb_filename_for_df.exists(), (
            f"{pdb_filename_for_df} does not exist."
            " Remember to run the preprocessing step."
        )

        dataset_structure = ResidueEnvironmentsDataset(
            [str(pdb_filename_for_df)], transformer=None
        )

        resenv_dataset = {}
        for resenv in dataset_structure:
            key = f"--{'CUSTOM'}--{resenv.chain_id}--{resenv.pdb_residue_number}--{index_to_one(resenv.restype_index)}--"
            resenv_dataset[key] = resenv

        df_structure_no_mt = pd.DataFrame.from_dict(
            resenv_dataset, orient="index", columns=["resenv"]
        )
        df_structure_no_mt.reset_index(inplace=True)
        df_structure_no_mt["index"] = df_structure_no_mt["index"].astype(str)
        res_info = pd.DataFrame(
            df_structure_no_mt["index"].str.split("--").tolist(),
            columns=["blank", "pdb_id", "chain_id", "pos", "wt_AA", "blank2"],
        )

        df_structure_no_mt["pdbid"] = res_info["pdb_id"]
        df_structure_no_mt["chainid"] = res_info["chain_id"]
        df_structure_no_mt["variant"] = res_info["wt_AA"] + res_info["pos"] + "X"
        aa_list = self.alphabet

        # Considering ALL single mutations in ALL sites.
        wildtype_residue_string = "".join(
            [v[0] for v in df_structure_no_mt["variant"].values]
        )

        df_structure = pd.DataFrame(
            df_structure_no_mt.values.repeat(20, axis=0),
            columns=df_structure_no_mt.columns,
        )
        pos_of_variant_column = df_structure.columns.get_loc("variant")
        for i in range(0, len(df_structure), 20):
            for j in range(20):
                df_structure.iloc[i + j, pos_of_variant_column] = (
                    df_structure.iloc[i + j, :]["variant"][:-1] + aa_list[j]
                )

        # This is a silly and inefficient way of doing this.
        # We should instead only build df_structure to have
        # the relevant variants from the start. That way
        # we avoid the above two-for-loops. But they're
        # practically instantaneous, so it's not a big deal.
        # (O(nm) doesn't matter all that much if n < 1000
        # , m is always 20, and the cost of each operation
        # is negligible)
        df_structure["mutant_residue_string"] = [""] * len(df_structure)
        if mutant_residue_strings is not None:
            # Compute the mutations associated to all strings in
            # mutant_residue_strings.

            # First, the following function computes where
            # the mutations are between the wildtype and
            # the mutant residue strings.
            # These are in the format ("replace", index_in_wildtype, index_in_mutant)
            mutations_in_rasp_format = []
            for mutant_residue_string in mutant_residue_strings:
                if mutant_residue_string == wildtype_residue_string:
                    # Then we append a mock mutation.
                    original_residue_as_single_character = wildtype_residue_string[0]
                    position_in_chain = res_info.iloc[0]["pos"]
                    mutant_residue_as_single_character = wildtype_residue_string[0]

                    mutation_in_rasp_format = (
                        original_residue_as_single_character
                        + f"{position_in_chain}"
                        + mutant_residue_as_single_character
                    )

                    mutations_in_rasp_format.append(mutation_in_rasp_format)

                    mask_for_this_mutation = df_structure["variant"].str.startswith(
                        mutation_in_rasp_format
                    )

                    df_structure.loc[
                        mask_for_this_mutation, "mutant_residue_string"
                    ] = mutant_residue_string
                    continue

                edits_ = edits_between_strings(
                    wildtype_residue_string, mutant_residue_string
                )
                for edit_ in edits_:
                    _, i, _ = edit_
                    original_residue_as_single_character = wildtype_residue_string[i]
                    position_in_chain = res_info.iloc[i]["pos"]
                    mutant_residue_as_single_character = mutant_residue_string[i]
                    mutation_in_rasp_format = (
                        original_residue_as_single_character
                        + f"{position_in_chain}"
                        + mutant_residue_as_single_character
                    )

                    mutations_in_rasp_format.append(mutation_in_rasp_format)

                    # Add the mutant residue string to the dataframe.
                    mask_for_this_mutation = df_structure["variant"].str.startswith(
                        mutation_in_rasp_format
                    )

                    df_structure.loc[
                        mask_for_this_mutation, "mutant_residue_string"
                    ] = mutant_residue_string

            # Filter df_structure to only contain the mutations in
            # mutations_in_rasp_format. (i.e. we need to focus on
            # only some of the positions, which are in the middle
            # of each string in mutations_in_rasp_format).
            df_structure = df_structure[
                df_structure["variant"].str.startswith(
                    tuple(set(mutations_in_rasp_format))
                )
            ]

            # We should attach the original mutant string to each
            # row in df_structure.

        df_structure.drop(columns="index", inplace=True)

        # Load PDB amino acid frequencies used to approximate unfolded states
        THIS_DIR = Path(__file__).parent.resolve()
        pdb_nlfs = -np.log(np.load(THIS_DIR / "pdb_frequencies.npz")["frequencies"])

        # # Add wt and mt idxs and freqs to df

        df_structure["wt_idx"] = df_structure.apply(
            lambda row: one_to_index(row["variant"][0]), axis=1
        )
        df_structure["mt_idx"] = df_structure.apply(
            lambda row: one_to_index(row["variant"][-1]), axis=1
        )
        df_structure["wt_nlf"] = df_structure.apply(
            lambda row: pdb_nlfs[row["wt_idx"]], axis=1
        )
        df_structure["mt_nlf"] = df_structure.apply(
            lambda row: pdb_nlfs[row["mt_idx"]], axis=1
        )

        return df_structure
