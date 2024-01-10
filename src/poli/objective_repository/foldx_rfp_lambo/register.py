"""RFP objective factory and black box function."""
__author__ = "Simon Bartels"

import logging
import yaml
from pathlib import Path
import os
from collections import namedtuple
import random
from typing import Tuple

import numpy as np
import hydra
import torch
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation
from poli.objective_repository.foldx_rfp_lambo import PROBLEM_SEQ, CORRECT_SEQ

import lambo
from lambo.tasks.proxy_rfp.proxy_rfp import ProxyRFPTask
from lambo.utils import AMINO_ACIDS
from lambo.utils import RESIDUE_ALPHABET
from lambo import __file__ as project_root_file


project_root = os.path.dirname(os.path.dirname(project_root_file))


class RFPWrapper(AbstractBlackBox):
    def __init__(
        self,
        task: ProxyRFPTask,
        base_candidates,
        parallelize: bool = False,
        num_workers: int = None,
        batch_size: int = None,
        evaluation_budget: int = float("inf"),
    ):
        super().__init__(
            RFPWrapperFactory().get_setup_information(),
            parallelize=parallelize,
            num_workers=num_workers,
            batch_size=batch_size,
            evaluation_budget=evaluation_budget,
        )
        self.task = task
        self.base_candidates = base_candidates
        self.sequences_aligned = False
        self.inverse_alphabet = {i + 1: AMINO_ACIDS[i] for i in range(len(AMINO_ACIDS))}
        self.inverse_alphabet[0] = "-"

    def _black_box(self, x, context=None):
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


class RFPWrapperFactory(AbstractProblemFactory):
    def __init__(self):
        self.alphabet = AMINO_ACIDS
        self.problem_sequence = PROBLEM_SEQ
        self.correct_sequence = CORRECT_SEQ

    def get_setup_information(self) -> ProblemSetupInformation:
        return ProblemSetupInformation("foldx_rfp_lambo", 244, False, self.alphabet)

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        config = get_config()

        # TODO: allow for bigger batch_sizes
        # For now (and because of the way the black box is implemented)
        # we only allow for batch_size=1
        if batch_size is None:
            batch_size = 1
        else:
            assert batch_size == 1

        # make problem reproducible
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

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
            all_seqs[
                np.where(all_seqs == self.problem_sequence)
            ] = self.correct_sequence
        return (
            RFPWrapper(
                bb_task,
                base_candidates,
                parallelize=parallelize,
                num_workers=num_workers,
                batch_size=batch_size,
                evaluation_budget=evaluation_budget,
            ),
            all_seqs,
            all_targets,
        )


tokenizer = {"_target_": "lambo.utils.ResidueTokenizer"}
Config = namedtuple("config", ["task", "tokenizer", "log_dir", "job_name", "timestamp"])


def get_config() -> Config:
    """
    Utility function with lambo specifc config to RFP task.
    """
    task = yaml.safe_load(
        Path(
            str(Path(lambo.__file__).parent.resolve().parent.resolve())
            + os.path.sep
            + "hydra_config"
            + os.path.sep
            + "task"
            + os.path.sep
            + "proxy_rfp.yaml"
        ).read_text()
    )
    config = Config(
        task,
        tokenizer,
        log_dir="data/experiments/test",
        job_name="null",
        timestamp="timestamp",
    )
    return config


if __name__ == "__main__":
    from poli.core.registry import register_problem

    rfp_problem_factory = RFPWrapperFactory()
    register_problem(
        rfp_problem_factory,
        conda_environment_name="poli__lambo",
        force=True,
    )
