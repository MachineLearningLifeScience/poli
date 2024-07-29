from pathlib import Path
from typing import Literal
from warnings import warn

import numpy as np
import pandas as pd
import torch

from poli.core.abstract_isolated_function import AbstractIsolatedFunction
from poli.core.black_box_information import BlackBoxInformation
from poli.objective_repository.gfp_cbas.cbas_alphabet_preprocessing import (
    convert_aas_to_idx_array,
    one_hot_encode_aa_array,
)
from poli.objective_repository.gfp_cbas.cbas_wrapper import CBASVAEWrapper
from poli.objective_repository.gfp_cbas.gfp_gp import SequenceGP


class GFPCBasIsolatedLogic(AbstractIsolatedFunction):
    def __init__(
        self,
        problem_type: Literal["gp", "vae", "elbo"],
        info: BlackBoxInformation,
        n_starting_points: int = 1,
        seed: int = None,
        functional_only: bool = False,
        ignore_stops: bool = True,
        unique=True,
    ):
        gfp_path = Path(__file__).parent.resolve() / "assets"
        self.info = info
        self.vae = CBASVAEWrapper(AA=len(info.alphabet), L=info.max_sequence_length).vae
        self.seed = seed

        data_df = pd.read_csv(gfp_path / "gfp_data.csv")[
            ["medianBrightness", "std", "nucSequence", "aaSequence"]
        ]
        gfp_wt_seq = np.atleast_1d(
            np.loadtxt(gfp_path / "avGFP_reference_sequence.fa", skiprows=1, dtype=str)
        )[0]
        self.reference_entry = data_df[
            data_df.nucSequence.str.lower() == gfp_wt_seq.lower()
        ]
        data_df = data_df.drop(
            self.reference_entry.index
        )  # take out WT reference sequence
        if unique:
            data_df = data_df.drop_duplicates(subset="aaSequence")
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
        if problem_type == "gp":
            gp_file_prefix = (
                Path(__file__).parent.resolve() / "assets" / "models" / "gp" / "gfp_gp"
            )
            self.model = SequenceGP(load=True, load_path_prefix=gp_file_prefix)
            self.f = self._seq_gp_predict
        elif problem_type == "elbo":
            self.model = self.vae.vae_
            self.f = self._elbo_predict
        elif problem_type == "vae":
            self.model = self.vae.encoder_
            self.f = self._vae_embedding
        else:
            raise NotImplementedError(
                f"Misspecified info: {info.name} does not contain [gp ; elbo ; vae]!"
            )
        self.data_df = data_df.loc[idx]

        x0 = np.array(list(self.reference_entry.aaSequence.values[0]))[
            np.newaxis, :
        ]  # WT reference sequence
        if (
            n_starting_points > 1
        ):  # take a random subset of available AA sequence at request
            _x0 = np.array(
                [list(s) for s in self.data_df.aaSequence[: n_starting_points - 1]]
            )
            x0 = np.concatenate([x0, _x0])
            assert x0.shape[0] == n_starting_points

        self.x0 = x0

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
        return -model_evaluation[:, 0].reshape(-1, 1)  # NOTE: minimize ELBO as target

    def _vae_embedding(self, x: np.ndarray) -> np.ndarray:
        """
        One hot encode sequence and VAE embed
        """
        assert self.model and self.model.__class__.__name__ == "Functional"
        oh_x = one_hot_encode_aa_array(x)
        return self.model.predict(oh_x)[0]

    def __call__(self, x: np.array, context=None) -> np.ndarray:
        """
        x is encoded sequence return function value given problem name
        """
        with torch.no_grad():
            cbas_mu = self.f(x)
        return np.array(cbas_mu)

    def __iter__(self, *args, **kwargs):
        warn(f"{self.__class__.__name__} iteration invoked. Not implemented!")


if __name__ == "__main__":
    from poli.core.registry import register_isolated_function

    register_isolated_function(
        GFPCBasIsolatedLogic,
        name="gfp_cbas__isolated",
        conda_environment_name="poli__protein_cbas",
    )
