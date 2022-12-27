from core.util.abstract_logger import AbstractLogger


class WandBLogger(AbstractLogger):
    def __init__(self):
        self.wandb = None
        self.verbose = True

    def initialize_logger(self, problem_factory, method_factory) -> str:
        import wandb
        self.wandb = wandb
        wandb.login(key="0b1dfca4b76cc7c1ac090500f702774c90837e5a")
        r: wandb.apis.public.Run = wandb.init(
            project=problem_factory.get_setup_information().get_problem_name(),
            notes="",
            tags=[],
            config=method_factory.get_params()
        )
        return r.storage_id

    def log(self, metrics: dict, step: int):
        from wandb import log as log_
        if self.verbose:
            for k in metrics.keys():
                print('\033[35m' + f"{k}: {metrics[k]}" + '\033[0m')
        log_(metrics)

    def finish(self) -> None:
        self.wandb.finish()
