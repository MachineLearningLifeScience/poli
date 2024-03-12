from poli.core.abstract_black_box import AbstractBlackBox


class IsolatedBlackBox(AbstractBlackBox):
    def __init__(
        self,
        name: str = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        **kwargs_for_black_box,
    ):

        super().__init__(
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            evaluation_budget=evaluation_budget,
        )
