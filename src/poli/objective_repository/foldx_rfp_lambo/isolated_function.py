"""RFP objective factory and black box function."""

__author__ = "Simon Bartels"

import logging
import os
from collections import namedtuple
from pathlib import Path

import hydra
import lambo
import numpy as np
import yaml
from lambo import __file__ as project_root_file
from lambo.utils import AMINO_ACIDS

from poli.core.abstract_isolated_function import AbstractIsolatedFunction
from poli.core.registry import register_isolated_function
from poli.core.util.files.download_files_from_github import (
    download_file_from_github_repository,
)
from poli.objective_repository.foldx_rfp_lambo import CORRECT_SEQ, PROBLEM_SEQ

project_root = os.path.dirname(os.path.dirname(project_root_file))
LAMBO_IN_POLI_OBJECTIVES_PATH = Path.home() / ".poli_objectives" / "lambo"
LAMBO_IN_POLI_OBJECTIVES_PATH.mkdir(parents=True, exist_ok=True)

LAMBO_PACKAGE_ROOT = Path(lambo.__file__).parent.resolve()

LAMBO_FOLDX_ASSETS_PDBS = [
    "1uis_A",
    "1xqm_A",
    "1yzw_A",
    "1zgo_A",
    "1ztu_A",
    "2h5q_A",
    "2h5r_A",
    "2qlg_B",
    "2v4e_B",
    "2vad_A",
    "2vae_A",
    "2vvh_C",
    "3bxa_A",
    "3e5v_A",
    "3gb3_A",
    "3ip2_A",
    "3m22_A",
    "3ned_A",
    "3nez_A",
    "3nt3_C",
    "3nt9_A",
    "3p8u_C",
    "3pj5_A",
    "3rwa_A",
    "3s05_B",
    "3s7q_A",
    "3u0k_A",
    "3u0l_A",
    "3u8c_A",
    "4cqh_A",
    "4edo_A",
    "4eds_A",
    "4h3l_A",
    "4hq8_A",
    "4jf9_B",
    "4kge_A",
    "4oqw_A",
    "4p76_A",
    "4q7t_B",
    "4q7u_A",
    "5ajg_A",
    "5dtl_A",
    "5ez2_A",
    "5jva_A",
    "5lk4_A",
    "6aa7_A",
    "6dej_A",
    "6fzn_A",
    "6mgh_D",
    "6xwy_A",
]

tokenizer = {"_target_": "lambo.utils.ResidueTokenizer"}
Config = namedtuple("config", ["task", "tokenizer", "log_dir", "job_name", "timestamp"])


def get_config() -> Config:
    """
    Utility function with lambo specifc config to RFP task.
    """
    task = yaml.safe_load(
        (LAMBO_IN_POLI_OBJECTIVES_PATH / "proxy_rfp.yaml").read_text()
        # Path(
        #     str(Path(lambo.__file__).parent.resolve().parent.resolve())
        #     + os.path.sep
        #     + "hydra_config"
        #     + os.path.sep
        #     + "task"
        #     + os.path.sep
        #     + "proxy_rfp.yaml"
        # ).read_text()
    )
    config = Config(
        task,
        tokenizer,
        log_dir="data/experiments/test",
        job_name="null",
        timestamp="timestamp",
    )
    return config


def _download_assets_from_lambo():
    if os.environ.get("GITHUB_TOKEN_FOR_POLI") is None:
        logging.warning(
            "This black box objective function require downloading files "
            "from GitHub. Since the API rate limit is 60 requests per hour, "
            "we recommend creating a GitHub token and setting it as an "
            "environment variable called GITHUB_TOKEN_FOR_POLI. "
            "To create a GitHub token like this, follow the instructions here: "
            "https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token"
        )

    # Downloading the config file into ~/.poli_objectives/lambo
    if not (LAMBO_IN_POLI_OBJECTIVES_PATH / "proxy_rfp.yaml").exists():
        download_file_from_github_repository(
            "samuelstanton/lambo",
            "hydra_config/task/proxy_rfp.yaml",
            str(Path.home() / ".poli_objectives" / "lambo" / "proxy_rfp.yaml"),
            commit_sha="431b052ad0e54a1ba4519272725310127c6377f1",
            parent_folders_exist_ok=True,
        )

    if not (
        LAMBO_PACKAGE_ROOT / "assets" / "fpbase" / "rfp_known_structures.csv"
    ).exists():
        download_file_from_github_repository(
            "samuelstanton/lambo",
            "lambo/assets/fpbase/rfp_known_structures.csv",
            str(LAMBO_PACKAGE_ROOT / "assets" / "fpbase" / "rfp_known_structures.csv"),
            commit_sha="431b052ad0e54a1ba4519272725310127c6377f1",
            parent_folders_exist_ok=True,
            verbose=True,
        )

    #     - proxy_rfp_seed_data.csv
    if not (
        LAMBO_PACKAGE_ROOT / "assets" / "fpbase" / "proxy_rfp_seed_data.csv"
    ).exists():
        download_file_from_github_repository(
            "samuelstanton/lambo",
            "lambo/assets/fpbase/proxy_rfp_seed_data.csv",
            str(LAMBO_PACKAGE_ROOT / "assets" / "fpbase" / "proxy_rfp_seed_data.csv"),
            commit_sha="431b052ad0e54a1ba4519272725310127c6377f1",
            parent_folders_exist_ok=True,
            verbose=True,
        )

    # - the sequences in the foldx folder.
    for folder_name in LAMBO_FOLDX_ASSETS_PDBS:
        if not (LAMBO_PACKAGE_ROOT / "assets" / "foldx" / folder_name).exists():
            download_file_from_github_repository(
                "samuelstanton/lambo",
                f"lambo/assets/foldx/{folder_name}",
                str(LAMBO_PACKAGE_ROOT / "assets" / "foldx" / folder_name),
                commit_sha="431b052ad0e54a1ba4519272725310127c6377f1",
                parent_folders_exist_ok=True,
                verbose=True,
            )


class RFPWrapperIsolatedLogic(AbstractIsolatedFunction):
    def __init__(
        self,
        seed: int = None,
    ):
        self.alphabet = AMINO_ACIDS
        self.problem_sequence = PROBLEM_SEQ
        self.correct_sequence = CORRECT_SEQ

        _download_assets_from_lambo()
        config = get_config()
        tokenizer = hydra.utils.instantiate(config.tokenizer)
        # NOTE: the task at this point is the original proxy rfp task
        bb_task = hydra.utils.instantiate(
            config.task, tokenizer=tokenizer, candidate_pool=[], seed=seed
        )
        base_candidates, base_targets, all_seqs, all_targets = bb_task.task_setup(
            config, project_root=project_root
        )

        # to ensure that algorithms will only request valid sequences, we provide the base sequences first
        permutation = np.arange(all_seqs.shape[0])
        base_candidate_idx = np.array([0, 3, 4, 16, 37, 39])
        permutation[base_candidate_idx] = np.arange(base_candidate_idx.shape[0])
        permutation[: base_candidate_idx.shape[0]] = base_candidate_idx
        all_seqs = all_seqs[permutation, ...]
        all_targets = all_targets[permutation, ...]
        assert np.all(all_targets[: base_targets.shape[0], ...] == base_targets)

        # all_seqs should be of shape [b, L], adding empty padding
        max_seq_length = max([len(seq) for seq in all_seqs])
        all_seqs = np.array(
            [list(seq) + [""] * (max_seq_length - len(seq)) for seq in all_seqs]
        )
        if self.problem_sequence in all_seqs:
            # substitute erroneous sequence "X in position 159" with correct PDB fasta sequence
            all_seqs[np.where(all_seqs == self.problem_sequence)] = (
                self.correct_sequence
            )

        self.task = bb_task
        self.base_candidates = base_candidates
        self.sequences_aligned = False
        self.inverse_alphabet = {i + 1: AMINO_ACIDS[i] for i in range(len(AMINO_ACIDS))}
        self.inverse_alphabet[0] = "-"

        self.x0 = all_seqs
        self.y0 = all_targets

    def __call__(self, x, context=None):
        best_b_cand = None
        min_hd = np.infty  # Hamming distance of best_b_cand to x

        # TODO: this assumes a batch_size of 1. Is that clear in the docs?
        seq = "".join(x[0])  # take out the string from the np array
        for b_cand in self.base_candidates:
            b_seq = b_cand.mutant_residue_seq
            if b_seq is None:
                raise ValueError(
                    "Base candidates (from lambo's FoldedCandidates) is empty."
                    "\nThis usually happens when the internal foldx process fails."
                    "Verify your foldx installation and try again."
                )
            if len(b_seq) != len(seq):
                continue
            hd = np.sum([seq[i] != b_seq[i] for i in range(len(seq))])
            if hd < min_hd:
                min_hd = hd
                best_b_cand = b_cand
        if best_b_cand is None:
            for b_cand in self.base_candidates:
                logging.fatal(
                    "%i is a valid candidate length." % len(b_cand.mutant_residue_seq)
                )
            raise RuntimeError("Cannot evaluate candidate of length %i." % len(seq))
        return self.task.score(
            self.task.make_new_candidates(np.array([best_b_cand]), np.array([seq]))
        )


if __name__ == "__main__":
    # inner_function = RFPWrapperIsolatedLogic()

    register_isolated_function(
        RFPWrapperIsolatedLogic,
        name="foldx_rfp_lambo__isolated",
        conda_environment_name="poli__lambo",
    )
