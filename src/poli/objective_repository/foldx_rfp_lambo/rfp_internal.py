"""
Wrap task setup to set variables of the observer.
Extends basic lambo task: ProxyRFPTask
"""
import logging
import numpy as np
from poli.core.util.external_observer import ExternalObserver

from lambo.tasks.proxy_rfp.proxy_rfp import ProxyRFPTask
from poli.objective_repository.foldx_rfp_lambo import RFPWrapperFactory
from poli.objective_repository.foldx_rfp_lambo import ALGORITHM, STARTING_N, BATCH_SIZE

# TODO: integrate this into the task class
# NOTE: this has to run as a dedicated process. Otherwise observer does not work.
observer = ExternalObserver()


class PoliRFPInternal(ProxyRFPTask):
    def __init__(
        self,
        tokenizer,
        candidate_pool,
        obj_dim,
        seed=None,
        transform=lambda x: x,
        num_start_examples=None,
        **kwargs,
    ):
        super().__init__(
            tokenizer, candidate_pool, obj_dim, transform, num_start_examples, **kwargs
        )
        self.seed = seed

    def task_setup(self, config, project_root, *args, **kwargs):
        base_candidates, base_targets, all_seqs, all_targets = super().task_setup(
            config, project_root=project_root, *args, **kwargs
        )
        problem_information = RFPWrapperFactory().get_setup_information()
        try:
            seed = config.seed  # if in config we have lambo
        # NOTE: this is hydra loaded by Lambo internally, taken from trial_id OR if None randomly generated by upcycle scripting-tools
        except AttributeError:
            seed = self.seed  # if not in config, we have created with wrapper
        logging.info(f"Calling RFPInternal, set seed={seed}")
        observer.initialize_observer(
            problem_information,
            {
                ALGORITHM: "LAMBO",
                STARTING_N: self.num_start_examples,
            },
            all_seqs,
            all_targets,
            seed,
        )
        return base_candidates, base_targets, all_seqs, all_targets

    def score(self, candidates):
        y = super().score(candidates)
        for i in range(len(candidates)):
            observer.observe(
                np.array([candidates[i].mutant_residue_seq]), y[i : i + 1, ...]
            )
        return y
