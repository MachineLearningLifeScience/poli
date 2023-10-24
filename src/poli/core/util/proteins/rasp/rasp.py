from typing import List
from pathlib import Path
import subprocess
import os, stat
import subprocess

import pandas as pd
import numpy as np

import torch

from Bio.PDB.Polypeptide import index_to_one, one_to_index

from .inner_rasp.cavity_model import (
    CavityModel,
    DownstreamModel,
    ResidueEnvironmentsDataset,
)

from .inner_rasp.helpers import (
    init_lin_weights,
    ds_pred,
    cavity_to_prism,
    get_seq_from_variant,
)
from .inner_rasp.pdb_parser_scripts.clean_pdb import (
    clean_pdb,
)
from .inner_rasp.pdb_parser_scripts.extract_environments import (
    extract_environments,
)

from poli.core.util.proteins.mutations import edits_between_strings

THIS_DIR = Path(__file__).parent.resolve()
HOME_DIR = THIS_DIR.home()
RASP_DIR = HOME_DIR / ".poli_objectives" / "rasp"
RASP_DIR.mkdir(parents=True, exist_ok=True)


def load_cavity_and_downstream_models(DEVICE: str = "cpu"):
    # DEVICE = "cpu"

    # TODO: Ask why this was implemented this way.
    # A transparent alternative would be to simply
    # load the model from the path itself.
    best_cavity_model_path = RASP_DIR / "cavity_model_15.pt"
    cavity_model_net = CavityModel(get_latent=True).to(DEVICE)
    cavity_model_net.load_state_dict(
        torch.load(f"{best_cavity_model_path}", map_location=DEVICE)
    )
    cavity_model_net.eval()
    ds_model_net = DownstreamModel().to(DEVICE)
    ds_model_net.apply(init_lin_weights)
    ds_model_net.eval()

    return cavity_model_net, ds_model_net

    ...


class RaspInterface:
    def __init__(self, working_dir: Path) -> None:
        self.working_dir = working_dir

        # Making the appropriate folders:
        (self.working_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.working_dir / "cleaned").mkdir(parents=True, exist_ok=True)
        (self.working_dir / "parsed").mkdir(parents=True, exist_ok=True)
        (self.working_dir / "output").mkdir(parents=True, exist_ok=True)

        # Downloading reduce and compile it.
        self.reduce_executable_path = None
        self.get_and_compile_reduce()

        # Downloading the cavity and downstream models.
        # TODO: implement this. What's the best way of doing this?
        # Where should we store the models?
        # self.download_cavity_and_downstream_models()

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
            subprocess.run(
                "git clone https://github.com/rlabduke/reduce.git",
                cwd=RASP_DIR,
                shell=True,
                stdout=subprocess.DEVNULL,
            )

            # compile it.

            subprocess.run(
                "make",
                cwd=REDUCE_DIR,
                stdout=subprocess.DEVNULL,
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
        # df_total["b_factors"] = df_total.apply(lambda row: row["resenv"].b_factors, axis=1)
        df_total = df_total.drop("resenv", axis=1)
        print(
            f"{len(df_structure)-len(df_ml)} data points dropped when matching total data with ml predictions in: {dataset_key}."
        )

        # Parse output into separate files by pdb, print to PRISM format
        for pdbid in df_total["pdbid"].unique():
            df_pdb = df_total[df_total["pdbid"] == pdbid]
            for chainid in df_pdb["chainid"].unique():
                pred_outdir = self.working_dir / "output" / f"{dataset_key}"
                pred_outdir.mkdir(parents=True, exist_ok=True)
                pred_outfile = pred_outdir / f"cavity_pred_{pdbid}_{chainid}.csv"
                print(
                    f"Parsing predictions from pdb: {pdbid}{chainid} into {pred_outfile}"
                )
                df_chain = df_pdb[df_pdb["chainid"] == chainid]
                df_chain = df_chain.assign(pos=df_chain["variant"].str[1:-1])
                df_chain["pos"] = pd.to_numeric(df_chain["pos"])
                first_res_no = min(df_chain["pos"])
                df_chain = df_chain.assign(wt_AA=df_chain["variant"].str[0])
                df_chain = df_chain.assign(mt_AA=df_chain["variant"].str[-1])
                seq = get_seq_from_variant(df_chain)
                df_chain.to_csv(pred_outfile, index=False)
                prism_outfile = pred_outdir / f"prism_cavity_{pdbid}_{chainid}.txt"

                # if (AF_ID !=''):
                #   prism_outfile = f"/content/output/{dataset_key}/prism_cavity_{AF_ID}_{chainid}.txt"
                # elif (PDB_ID !=''):
                #   prism_outfile = f"/content/output/{dataset_key}/prism_cavity_{PDB_ID}_{chainid}.txt"
                # elif PDB_custom:
                #   prism_outfile = f"/content/output/{dataset_key}/prism_cavity_XXXX_{chainid}.txt"
                cavity_to_prism(df_chain, pdbid, chainid, seq, prism_outfile)

        return df_total

    def download_cavity_and_downstream_models(self):
        """
        This function downloads the cavity and downstream models
        at the ~/.poli_objectives/rasp directory.

        TODO: Find a way to download the models without
        having to clone the entire repo.
        """
        raise NotImplementedError

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
        pdb_tools_command_for_cleanup = [
            "pdb_selchain",
            f"-{chain}",
            str(wildtype_pdb_path.resolve()),
            "|",
            "pdb_delhetatm",
            "|",
            "pdb_delres",
            "--999:0:1",
            "|",
            "pdb_fixinsert",
            "|",
            "pdb_tidy",
            ">",
            str(raw_output_path.resolve()),
        ]

        subprocess.run(
            " ".join(pdb_tools_command_for_cleanup),
            check=True,
            cwd=self.working_dir,
            shell=True,
        )

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
        pdb_id = wildtype_pdb_path.stem.split("_")[0]
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
        aa_list = [
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

        # Considering ALL single mutations in ALL sites.
        wildtype_residue_string = "".join(
            [v[0] for v in df_structure_no_mt["variant"].values]
        )

        df_structure = pd.DataFrame(
            df_structure_no_mt.values.repeat(20, axis=0),
            columns=df_structure_no_mt.columns,
        )
        for i in range(0, len(df_structure), 20):
            for j in range(20):
                df_structure.iloc[i + j, :]["variant"] = (
                    df_structure.iloc[i + j, :]["variant"][:-1] + aa_list[j]
                )

        # This is a silly and inefficient way of doing this.
        # We should instead only build df_structure to have
        # the relevant variants from the start. That way
        # we avoid the above two-for-loops. But they're
        # practically instantaneous, so it's not a big deal.
        # (O(nm) doesn't matter all that much if n < 1000
        # and m is always 20)
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
                    mutations_in_rasp_format.append(
                        wildtype_residue_string[0] + f"{1}" + wildtype_residue_string[0]
                    )
                    continue

                edits_ = edits_between_strings(
                    wildtype_residue_string, mutant_residue_string
                )
                for edit_ in edits_:
                    _, i, _ = edit_
                    mutations_in_rasp_format.append(
                        wildtype_residue_string[i] + f"{i+1}" + mutant_residue_string[i]
                    )

            # Filter df_structure to only contain the mutations in
            # mutations_in_rasp_format. (i.e. we need to focus on
            # only some of the positions, which are in the middle
            # of each string in mutations_in_rasp_format).
            df_structure = df_structure[
                df_structure["variant"].str.startswith(
                    tuple(
                        set(
                            [
                                mutation_in_rasp_format
                                for mutation_in_rasp_format in mutations_in_rasp_format
                            ]
                        )
                    )
                )
            ]

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
