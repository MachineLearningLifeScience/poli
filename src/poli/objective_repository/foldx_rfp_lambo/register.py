__author__ = "Simon Bartels"

import logging
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

from lambo.tasks.proxy_rfp.proxy_rfp import ProxyRFPTask
from lambo.utils import AMINO_ACIDS
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
        # seq = "".join([self.inverse_alphabet[x[0, i]] for i in range(x.shape[1])])
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
        # raise NotImplementedError("TODO: adapt to new alphabet definition in Poli")
        # self.alphabet = {AMINO_ACIDS[i]: i+1 for i in range(len(AMINO_ACIDS))}
        # self.alphabet["-"] = 0
        # self.alphabet["X"] = 0  # TODO: is that the way I want to handle this stupid single sequence?
        self.alphabet = AMINO_ACIDS

    def get_setup_information(self) -> ProblemSetupInformation:
        return ProblemSetupInformation("FOLDX_RFP", 244, False, self.alphabet)

    def create(self, seed: int = 0) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
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
        return RFPWrapper(bb_task, base_candidates), all_seqs, all_targets


task = {
    "_target_": "lambo.tasks.proxy_rfp.proxy_rfp.ProxyRFPTask",
    "obj_dim": 2,
    "log_prefix": "proxy_rfp",
    "batch_size": 16,
    "max_len": 244,
    "max_num_edits:": None,
    "max_ngram_size": 1,
    "allow_len_change": False,
    "num_start_examples": 512,
}
tokenizer = {"_target_": "lambo.utils.ResidueTokenizer"}
Config = namedtuple("config", ["task", "tokenizer", "log_dir", "job_name", "timestamp"])

name_is_main = __name__ == "__main__"


# this is to trick Hydra...
# __name__ = '__main__'
# config_path = os.path.join(project_root, 'hydra_config')
def get_config():
    global conf
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
    # os.chdir(os.path.join(project_root, "scripts"))
    from poli.core.registry import register_problem

    rfp_problem_factory = RFPWrapperFactory()
    register_problem(
        rfp_problem_factory,
        conda_environment_name="lambo-env",  # TODO: poli__lambo
    )
