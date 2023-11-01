from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.util import batch
from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli.core.util.seeding import seed_numpy, seed_python


class GFPBlackBox(AbstractBlackBox):
    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        seed: int = None,
    ):
        gfp_df_path = Path(__file__).parent.resolve() / "assets" / "gfp_data.csv"
        self.batch_size = batch_size
        self.seed = seed
        self.gfp_lookup_df = pd.read_csv(gfp_df_path)[
            ["medianBrightness", "aaSequence"]
        ]
        super().__init__(info, batch_size, parallelize, num_workers)

    def _black_box(self, x: np.array, context=None) -> np.ndarray:
        """
        x is string sequence which we look-up in avilable df, return median Brightness
        """
        if isinstance(x, np.ndarray):
            _arr = x.copy()
            x = ["".join(_seq) for _seq in _arr]
        ys = []
        for _x in x:
            seq_subsets = self.gfp_lookup_df[
                self.gfp_lookup_df.aaSequence.str.lower() == _x.lower()
            ]
            # multiple matches possible, shuffle and return one:
            candidate = seq_subsets.sample(n=1, random_state=self.seed)
            ys.append(candidate.medianBrightness)
        return np.array(ys)


class GFPSelectionProblemFactory(AbstractProblemFactory):
    def get_setup_information(self) -> ProblemSetupInformation:
        """
        The problem is set up such that all available sequences
        are provided in x0, however only batch_size amount of observations are known.
        I.e. f(x0[:batch_size]) is returned as f_0 .
        The task is to find the minimum, given that only limited inquiries (batch_size) can be done.
        Given that all X are known it is recommended to use an acquisition function to rank
        and inquire the highest rated sequences with the _black_box.
        """
        problem_setup_info = ProblemSetupInformation(
            name="gfp_select",
            max_sequence_length=237,  # max len of aaSequence
            alphabet=AMINO_ACIDS,
            aligned=True,  # TODO: perhaps add the fact that there is a random state here?
        )
        return problem_setup_info

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        seed_numpy(seed)
        seed_python(seed)
        problem_info = self.get_setup_information()
        f = GFPBlackBox(
            info=problem_info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            seed=seed,
        )

        randomized_df = f.gfp_lookup_df.sample(frac=1, random_state=seed).reset_index()
        # create 2D array for blackbox evaluation
        x0 = np.array([list(_s) for _s in randomized_df.aaSequence.to_numpy()])
        f_0 = f(x0[:batch_size])

        return f, x0, f_0


if __name__ == "__main__":
    from poli.core.registry import register_problem

    gfp_problem_factory = GFPSelectionProblemFactory()
    register_problem(
        gfp_problem_factory,
        conda_environment_name="poli__protein",
    )
