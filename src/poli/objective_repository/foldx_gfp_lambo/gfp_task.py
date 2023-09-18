import hydra
import torch
from botorch.utils.multi_objective import pareto

from lambo.candidate import FoldedCandidate
from lambo.tasks.base_task import BaseTask


class ProxyGFPTask(BaseTask):
    def __init__(
        self,
        tokenizer,
        candidate_pool,
        obj_dim,
        transform=lambda x: x,
        num_start_examples=1024,
        batch_size=1,
        candidate_weights=None,
        max_len=None,
        max_ngram_size=1,
        allow_len_change=True,
        **kwargs
    ):
        super().__init__(
            tokenizer,
            candidate_pool,
            obj_dim,
            transform,
            batch_size,
            candidate_weights,
            max_len,
            max_ngram_size,
            allow_len_change,
            **kwargs
        )
        self.op_types = ["sub"]
        self.num_start_examples = num_start_examples
    
    def task_setup(self, config, project_root=None, *args, **kwargs):
        project_root = hydra.utils.get_original_cwd() if project_root is None else project_root
        candidate_data = pd.read_csv()