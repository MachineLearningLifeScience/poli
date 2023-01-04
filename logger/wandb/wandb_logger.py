import wandb
from wandb import log as log_

from core.util.abstract_logger import AbstractLogger


class WandBLogger(AbstractLogger):
    def __init__(self):
        self.verbose = True

    def initialize_logger(self, problem_setup_info, caller_info) -> str:
        wandb.login(key="0b1dfca4b76cc7c1ac090500f702774c90837e5a")
        r: wandb.apis.public.Run = wandb.init(
            project=problem_setup_info.get_problem_name(),
            notes="",
            tags=[],
            config=caller_info
        )
        return r.get_url()

    def log(self, metrics: dict, step: int):
        if self.verbose:
            for k in metrics.keys():
                print('\033[35m' + f"{k}: {metrics[k]}" + '\033[0m')
        log_(metrics)

    def finish(self) -> None:
        wandb.finish()
