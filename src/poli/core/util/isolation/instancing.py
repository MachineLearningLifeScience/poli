from .external_black_box import ExternalBlackBox


def __instance_as_isolated_process(
    name: str,
    seed: int = None,
    batch_size: int = None,
    parallelize: bool = False,
    num_workers: int = None,
    evaluation_budget: int = float("inf"),
    quiet: bool = False,
    **kwargs_for_black_box,
) -> ExternalBlackBox: ...
