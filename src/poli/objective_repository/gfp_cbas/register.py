from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.abstract_problem_factory import AbstractProblemFactory
from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.util import batch
from poli.core.util.proteins.defaults import AMINO_ACIDS
from poli.core.util.seeding import seed_numpy, seed_python
from poli.objective_repository.gfp_cbas.cbas_wrapper import CBASVAEWrapper
from poli.objective_repository.gfp_cbas.cbas_alphabet_preprocessing import (
    AA,
    get_gfp_X_y_aa,
    one_hot_encode_aa_array,
)


class GFPCBasBlackBox(AbstractBlackBox):
    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        seed: int = None,
        functional_only: bool = False,
        ignore_stops: bool = True,
    ):
        gfp_df_path = Path(__file__).parent.resolve() / "assets" / "gfp_data.csv"
        self.info = info
        self.vae = CBASVAEWrapper(AA=len(info.alphabet), L=info.max_sequence_length).vae
        self.batch_size = batch_size
        self.seed = seed
        data_df = pd.read_csv(gfp_df_path)
        if self.seed:  # if random seed is provided, shuffle the data
            data_df.sample(frac=1, random_state=seed).reset_index()
        if (
            functional_only
        ):  # if functional only setting, threshold by median Brightness
            idx = data_df.loc[
                (data_df["medianBrightness"] > data_df["medianBrightness"].mean())
            ].index
        else:
            idx = data_df.index
        data_df = data_df.loc[idx]
        if ignore_stops:  # ignore incorrect encodings
            idx = data_df.loc[~data_df["aaSequence"].str.contains("!")].index
        self.data_df = data_df.loc[idx]
        # self.X, self.y = get_gfp_X_y_aa(self.gfp_lookup_df)
        super().__init__(info, batch_size, parallelize, num_workers)

    def _black_box(self, x: np.array, context=None) -> np.ndarray:
        """
        x is encoded sequence which we encode, return VAE ELBO
        """
        oh_x = one_hot_encode_aa_array(x)
        with torch.no_grad():
            cbas_mu, cbas_log_var = self.vae.encoder_.predict(oh_x)
        return np.array(cbas_mu)


class GFPCBasProblemFactory(AbstractProblemFactory):
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
            name="gfp_cbas",
            max_sequence_length=237,  # max len of aaSequence
            alphabet=AA,
            aligned=True,  # TODO: perhaps add the fact that there is a random state here?
        )
        return problem_setup_info

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        x0_size: int = 128,  # TODO: this should go into problem info instead?
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        seed_numpy(seed)
        seed_python(seed)
        problem_info = self.get_setup_information()
        f = GFPCBasBlackBox(
            info=problem_info,
            batch_size=batch_size,
            parallelize=parallelize,
            num_workers=num_workers,
            seed=seed,
        )
        x0 = np.array([list(s) for s in f.data_df.iloc[:x0_size].aaSequence])
        f_0 = f(x0)

        return f, x0, f_0


if __name__ == "__main__":
    from poli.core.registry import register_problem

    gfp_problem_factory = GFPCBasProblemFactory()
    register_problem(
        gfp_problem_factory,
        conda_environment_name="poli__protein_cbas",
    )
