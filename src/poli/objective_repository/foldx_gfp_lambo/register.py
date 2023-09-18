__author__ = "Richard Michael"

import os
from typing import Tuple

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from lambo.utils import AMINO_ACIDS


# TODO: we might need a GFP Task for Lambo side, load, eval data!


class GFPWrapper(AbstractBlackBox):
    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
    ):
        super().__init__(info, batch_size, parallelize, num_workers)
        self.task


class GFPWrapperFactory:
    pass


if __name__ == "__main__":
    gfp_problem_factory = GFPWrapperFactory()
    register_problem(
        gfp_problem_factory,
        conda_environment_name="poli__lambo",  # TODO: environment that can load CBAS?
        force=True,
    )
