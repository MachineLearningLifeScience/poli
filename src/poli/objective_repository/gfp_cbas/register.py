from warnings import warn
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
    convert_aas_to_idx_array,
)
from poli.objective_repository.gfp_cbas.gfp_gp import SequenceGP


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
        data_df = pd.read_csv(gfp_df_path)[["medianBrightness", "std", "aaSequence"]]
        if self.seed:  # if random seed is provided, shuffle the data
            data_df = data_df.sample(frac=1, random_state=seed)
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
        self.model = None
        # use ProblemSetupInfo name to specify black-box function
        if "gp" in info.name.lower():
            gp_file_prefix = (
                Path(__file__).parent.resolve() / "assets" / "models" / "gp" / "gfp_gp"
            )
            self.model = SequenceGP(load=True, load_path_prefix=gp_file_prefix)
            self.f = self._seq_gp_predict
        elif "elbo" in info.name.lower():
            self.model = self.vae.vae_
            self.f = self._elbo_predict
        elif "vae" in info.name.lower():
            self.model = self.vae.encoder_
            self.f = self._vae_embedding
        else:
            raise NotImplementedError(
                f"Misspecified info: {info.name} does not contain [fluorescence ; elbo ; vae]!"
            )
        self.data_df = data_df.loc[idx]
        super().__init__(info, batch_size, parallelize, num_workers)

    def _seq_gp_predict(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate Listgarten GP on label-encoded sequence input.
        Return negative pred.mean of GP (minimization problem)
        """
        assert self.model and self.model.__class__.__name__ == "SequenceGP"
        le_x = convert_aas_to_idx_array(x)
        return -self.model.predict(le_x)  # NOTE: predict negative mu

    def _elbo_predict(self, x: np.ndarray) -> np.ndarray:
        """
        One Hot encode and VAE evaluate given reference zero point.
        Calls Keras engine function evaluation to compute ELBO.
        """
        assert self.model and self.model.__class__.__name__ == "Functional"
        oh_x = one_hot_encode_aa_array(x)
        # model.evaluate takes two array inputs: input , [decoder_out, KLD ref.]
        # TODO: batched evaluation not working: returns one value only (not size of batch) -> iterating - fix this for speed-up.
        kld_reference_prior = np.zeros([1, oh_x.shape[-1]])
        model_evaluation = np.array(
            [
                self.model.evaluate(
                    _oh_x_seq[np.newaxis, :],
                    [_oh_x_seq[np.newaxis, :], kld_reference_prior],
                    batch_size=1,
                    verbose=0,
                )
                for _oh_x_seq in oh_x
            ]
        )
        # subselect ELBO as first column:
        return model_evaluation[:, 0].reshape(-1, 1)

    def _vae_embedding(self, x: np.ndarray) -> np.ndarray:
        """
        One hot encode sequence and VAE embed
        """
        assert self.model and self.model.__class__.__name__ == "Functional"
        oh_x = one_hot_encode_aa_array(x)
        return self.model.predict(oh_x)[0]

    def _black_box(self, x: np.array, context=None) -> np.ndarray:
        """
        x is encoded sequence return function value given problem name
        """
        with torch.no_grad():
            cbas_mu = self.f(x)
        return np.array(cbas_mu)

    def __iter__(self, *args, **kwargs):
        warn(f"{self.__class__.__name__} iteration invoked. Not implemented!")


class GFPCBasProblemFactory(AbstractProblemFactory):
    def __init__(self, problem_type: str = "gp") -> None:
        super().__init__()
        if problem_type.lower() not in ["gp", "vae", "elbo"]:
            raise NotImplementedError(
                f"Specified problem type: {problem_type} does not exist!"
            )
        self.problem_type = problem_type.lower()

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
            name=f"gfp_cbas_{self.problem_type}",
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
        evaluation_budget: int = 100000,
        n_starting_points: int = 128,
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        """
        Seed value required to shuffle the data, otherwise CSV asset data index unchanged.
        """
        if problem_type.lower() not in ["gp", "vae", "elbo"]:
            raise NotImplementedError(
                f"Specified problem type: {problem_type} does not exist!"
            )
        self.problem_type = problem_type  # required in class scope for setup info
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
        x0 = np.array([list(s) for s in f.data_df.iloc[:n_starting_points].aaSequence])
        f_0 = f(x0)

        return f, x0, f_0


if __name__ == "__main__":
    from poli.core.registry import register_problem

    gfp_problem_factory = GFPCBasProblemFactory()
    register_problem(
        gfp_problem_factory,
        conda_environment_name="poli__protein_cbas",
    )
    gfp_problem_factory.create(seed=12)
    # instantiate different types of CBas problems:
    gfp_problem_factory_vae = GFPCBasProblemFactory(problem_type="vae")
    register_problem(
        gfp_problem_factory_vae,
        conda_environment_name="poli__protein_cbas",
    )
    gfp_problem_factory_vae.create(seed=12)
    gfp_problem_factory_elbo = GFPCBasProblemFactory(problem_type="elbo")
    register_problem(
        gfp_problem_factory_elbo,
        conda_environment_name="poli__protein_cbas",
    )
    # gfp_problem_factory_elbo.create(seed=12)
