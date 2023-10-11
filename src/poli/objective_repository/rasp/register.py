import glob
import os
import pandas as pd
import torch

from Bio.PDB.Polypeptide import index_to_one, one_to_index

from poli.objective_repository.rasp.inner_rasp.rasp_model import (
    CavityModel,
    DownstreamModel,
    ResidueEnvironmentsDataset,
)

from poli.objective_repository.rasp.inner_rasp.helpers import (
    init_lin_weights,
    ds_pred,
    rasp_to_prism,
    get_seq_from_variant,
)

NUM_ENSEMBLE = 10
TASK_ID = int(1)
PER_TASK = int(1)

AF_ID = ""
PDB_ID = ""


def create_df_structure():
    pdb_filenames_ds = sorted(
        glob.glob(f"/content/data/test/predictions/parsed/*coord*")
    )

    dataset_structure = ResidueEnvironmentsDataset(pdb_filenames_ds, transformer=None)

    resenv_dataset = {}
    for resenv in dataset_structure:
        if AF_ID != "":
            key = f"--{AF_ID}--{resenv.chain_id}--{resenv.pdb_residue_number}--{index_to_one(resenv.restype_index)}--"
        elif PDB_ID != "":
            key = f"--{PDB_ID}--{resenv.chain_id}--{resenv.pdb_residue_number}--{index_to_one(resenv.restype_index)}--"
        else:
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
    df_structure = pd.DataFrame(
        df_structure_no_mt.values.repeat(20, axis=0), columns=df_structure_no_mt.columns
    )
    for i in range(0, len(df_structure), 20):
        for j in range(20):
            df_structure.iloc[i + j, :]["variant"] = (
                df_structure.iloc[i + j, :]["variant"][:-1] + aa_list[j]
            )

    df_structure.drop(columns="index", inplace=True)

    return df_structure


def load_cavity_and_downstream_models():
    DEVICE = "cpu"

    # TODO: Ask why this was implemented this way.
    # A transparent alternative would be to simply
    # load the model from the path itself.
    best_cavity_model_path = open(
        f"/content/output/cavity_models/best_model_path.txt", "r"
    ).read()
    cavity_model_net = CavityModel(get_latent=True).to(DEVICE)
    cavity_model_net.load_state_dict(torch.load(f"{best_cavity_model_path}"))
    cavity_model_net.eval()
    ds_model_net = DownstreamModel().to(DEVICE)
    ds_model_net.apply(init_lin_weights)
    ds_model_net.eval()

    return cavity_model_net, ds_model_net


def predict(
    cavity_model_net, ds_model_net, df_structure, dataset_key, NUM_ENSEMBLE, DEVICE
):
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
    df_total = df_total.drop("resenv", 1)
    print(
        f"{len(df_structure)-len(df_ml)} data points dropped when matching total data with ml predictions in: {dataset_key}."
    )

    # Parse output into separate files by pdb, print to PRISM format
    for pdbid in df_total["pdbid"].unique():
        df_pdb = df_total[df_total["pdbid"] == pdbid]
        for chainid in df_pdb["chainid"].unique():
            pred_outfile = (
                f"{os.getcwd()}/output/{dataset_key}/cavity_pred_{pdbid}_{chainid}.csv"
            )
            print(f"Parsing predictions from pdb: {pdbid}{chainid} into {pred_outfile}")
            df_chain = df_pdb[df_pdb["chainid"] == chainid]
            df_chain = df_chain.assign(pos=df_chain["variant"].str[1:-1])
            df_chain["pos"] = pd.to_numeric(df_chain["pos"])
            first_res_no = min(df_chain["pos"])
            df_chain = df_chain.assign(wt_AA=df_chain["variant"].str[0])
            df_chain = df_chain.assign(mt_AA=df_chain["variant"].str[-1])
            seq = get_seq_from_variant(df_chain)
            df_chain.to_csv(pred_outfile, index=False)
            prism_outfile = (
                f"/content/output/{dataset_key}/prism_cavity_{pdbid}_{chainid}.txt"
            )

            # if (AF_ID !=''):
            #   prism_outfile = f"/content/output/{dataset_key}/prism_cavity_{AF_ID}_{chainid}.txt"
            # elif (PDB_ID !=''):
            #   prism_outfile = f"/content/output/{dataset_key}/prism_cavity_{PDB_ID}_{chainid}.txt"
            # elif PDB_custom:
            #   prism_outfile = f"/content/output/{dataset_key}/prism_cavity_XXXX_{chainid}.txt"
            rasp_to_prism(df_chain, pdbid, chainid, seq, prism_outfile)
    ...


if __name__ == "__main__":
    wildtype_pdb_path = ...
    chain_to_keep = ...

    # Command to run to clean up the chains:
    # This one will need pdb_selchain, pdb_delhetatm, pdb_delres, pdb_fixinsert, and pdb_tidy.
    # These are all available in the pdb-tools package, which
    # is already installed in the conda environment.
    # !pdb_selchain -"$chain" /content/query_protein.pdb | pdb_delhetatm | pdb_delres --999:0:1 | pdb_fixinsert | pdb_tidy  > /content/data/test/predictions/raw/query_protein_uniquechain.pdb

    # Commands required to pre-process the PDBs:
    # These ones will need clean_pdb.py, reduce, and extract_environments.py
    # !python3 /content/src/pdb_parser_scripts/clean_pdb.py --pdb_file_in /content/data/test/predictions/raw/query_protein_uniquechain.pdb --out_dir /content/data/test/predictions/cleaned/ --reduce_exe /content/src/pdb_parser_scripts/reduce/reduce #&> /dev/null
    # !python3 /content/src/pdb_parser_scripts/extract_environments.py --pdb_in /content/data/test/predictions/cleaned/query_protein_uniquechain_clean.pdb  --out_dir /content/data/test/predictions/parsed/  #&> /dev/null

    # The output of these is a .npz file in the parsed directory.
    # More precisely, it generates:
    # "/content/data/test/predictions/parsed/query_protein_uniquechain_clean_coordinate_features.npz"

    ...
