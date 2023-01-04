import wandb
from wandb.sdk.lib import RunDisabled

from core.util.abstract_logger import AbstractLogger

wandb.login(key="0b1dfca4b76cc7c1ac090500f702774c90837e5a")


class WandBLogger(AbstractLogger):
    def __init__(self):
        self.verbose = True

    def initialize_logger(self, problem_setup_info, caller_info) -> str:
        r: wandb.apis.public.Run = wandb.init(
            project=problem_setup_info.get_problem_name(),
            notes="",
            tags=[],
            config=caller_info
        )
        if type(r) is RunDisabled:
            return None
        return r.path

    def log(self, metrics: dict, step: int):
        if self.verbose:
            for k in metrics.keys():
                print('\033[35m' + f"{k}: {metrics[k]}" + '\033[0m')
        wandb.log(metrics, step=step)

    def finish(self) -> None:
        wandb.finish()
