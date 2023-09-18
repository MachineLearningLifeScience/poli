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

from lambo.tasks.proxy_rfp.proxy_rfp import ProxyRFPTask
from lambo.utils import AMINO_ACIDS
from lambo.utils import RESIDUE_ALPHABET
from lambo import __file__ as project_root_file


project_root = os.path.dirname(os.path.dirname(project_root_file))
conf = None


class RFPWrapper(AbstractBlackBox):
    def __init__(self, task: ProxyRFPTask, base_candidates):
        super().__init__(RFPWrapperFactory().get_setup_information())
        self.task = task
        self.base_candidates = base_candidates
        self.sequences_aligned = False
        self.inverse_alphabet = {i + 1: AMINO_ACIDS[i] for i in range(len(AMINO_ACIDS))}
        self.inverse_alphabet[0] = "-"

    def _black_box(self, x, context=None):
        best_b_cand = None
        min_hd = np.infty  # Hamming distance of best_b_cand to x
        seq = x[0]  # take out the string from the np array
        for b_cand in self.base_candidates:
            b_seq = b_cand.mutant_residue_seq
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
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        config = get_config()
        config = conf
        # make problem reproducible
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        tokenizer = hydra.utils.instantiate(config.tokenizer)
        bb_task = hydra.utils.instantiate(
            config.task, tokenizer=tokenizer, candidate_pool=[]
        )
        base_candidates, base_targets, all_seqs, all_targets = bb_task.task_setup(
            config, project_root=project_root
        )
        if self.problem_sequence in all_seqs:
            # substitute erroneous sequence "X in position 159" with correct PDB fasta sequence
            all_seqs[
                np.where(all_seqs == self.problem_sequence)
            ] = self.correct_sequence
        return RFPWrapper(bb_task, base_candidates), all_seqs, all_targets


tokenizer = {"_target_": "lambo.utils.ResidueTokenizer"}
Config = namedtuple("config", ["task", "tokenizer", "log_dir", "job_name", "timestamp"])

name_is_main = __name__ == "__main__"


# satisfy Hydra
def get_config():
    global conf
    task = yaml.safe_load(
        Path(
            os.path.dirname(__file__) + os.path.sep + "lambo_task_config.yaml"
        ).read_text()
    )
    config = Config(
        task,
        tokenizer,
        log_dir="data/experiments/test",
        job_name="null",
        timestamp="timestamp",
    )
    conf = config
    return config


if name_is_main:
    from poli.core.registry import register_problem

    rfp_problem_factory = RFPWrapperFactory()
    register_problem(
        rfp_problem_factory,
        conda_environment_name="poli__lambo",
        force=True,
    )
