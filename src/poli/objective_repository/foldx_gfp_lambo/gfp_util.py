import os
import numpy as np
import pandas as pd
from pathlib import Path

from poli.core.util.proteins.foldx import FoldxInterface


def get_gfp_base_seq():
    """
    Returns GFP WT Sequence.
    # TODO: file does not exist under assets!
    """
    wt_fasta_filename = os.path.join(os.path.dirname(__file__), "assets", "avGFP_reference_sequence.fa")
    with open(wt_fasta_filename) as infile:
        fa_seq = infile.readlines()
    seq = fa_seq[1].strip()
    return seq


def read_gfp_data(path: str=None) -> pd.DataFrame:
    """
    Read the GFP data into pandas DataFrame.
    """
    gfp_tsv_filename = os.path.join(os.path.dirname(__file__), "assets", "gfp_data.csv")
    df = pd.read_csv(gfp_tsv_filename, delimiter="\t")
    return df


def get_mutations_against_wt(wt: str, seq: str) -> str:
    _wt = np.array(list(wt))
    _seq = np.array(list(seq))
    mut_idx = np.where(_wt != _seq)[0]
    mut_list = []
    for idx in mut_idx:
        # Note PDB indexing starts at 1!
        mut_str = "".join([_wt[idx], str(idx+1), _seq[idx]])
        mut_list.append(mut_str)
    return mut_list



# TODO: compute sasa and stability against reference GFP
def compute_sasa_and_stability_from_sequences_against_wt(working_dir: str, wt_pdb: str, sequences: pd.Series):
    stability_vals = []
    sasa_vals = []
    foldx_interface = FoldxInterface(Path(working_dir))
    for seq in sequences:
        _seq = avgfp_to_ref_gfp(seq)
        stability, sasa = foldx_interface.compute_stability_and_sasa(wt_pdb, [_seq])
        stability_vals.append(stability)
        sasa_vals.append(sasa)
    stability_vals, sasa_vals


def avgfp_to_ref_gfp(seq_str, ref_len=230):
    return "a" +  seq_str[:ref_len-1]