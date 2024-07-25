"""
This module implements LogP in _exactly_ the same was
as LaMBO does it [1]. We do this by importing the function
they use to compute `logp`.

References
----------
[1] “Accelerating Bayesian Optimization for Biological Sequence Design with Denoising Autoencoders.”
Stanton, Samuel, Wesley Maddox, Nate Gruver, Phillip Maffettone,
Emily Delaney, Peyton Greenside, and Andrew Gordon Wilson.
arXiv, July 12, 2022. http://arxiv.org/abs/2203.12742.
"""

import logging
import os
from pathlib import Path

import lambo
import numpy as np
from lambo import __file__ as project_root_file
from lambo.tasks.chem.logp import logP

from poli.core.abstract_isolated_function import AbstractIsolatedFunction
from poli.core.util.chemistry.string_to_molecule import translate_selfies_to_smiles
from poli.core.util.files.download_files_from_github import (
    download_file_from_github_repository,
)

project_root = os.path.dirname(os.path.dirname(project_root_file))
LAMBO_IN_POLI_OBJECTIVES_PATH = Path.home() / ".poli_objectives" / "lambo"
LAMBO_IN_POLI_OBJECTIVES_PATH.mkdir(parents=True, exist_ok=True)

LAMBO_PACKAGE_ROOT = Path(lambo.__file__).parent.resolve()


def _download_assets_from_lambo():
    if os.environ.get("GITHUB_TOKEN_FOR_POLI") is None:
        logging.warning(
            "This black box objective function require downloading files "
            "from GitHub. Since the API rate limit is 60 requests per hour, "
            "we recommend creating a GitHub token and setting it as an "
            "environment variable called GITHUB_TOKEN_FOR_POLI. "
            "To create a GitHub token like this, follow the instructions here: "
            "https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token"
        )

    # We should download files that are necessary, just like in
    # the other lambo objective function.

    # These files are lambo/tasks/chem/SA_Score/fpscores.pkl.gz
    fpscores_filepath = (
        LAMBO_PACKAGE_ROOT / "tasks" / "chem" / "SA_Score" / "fpscores.pkl.gz"
    )
    if not fpscores_filepath.exists():
        download_file_from_github_repository(
            "samuelstanton/lambo",
            "lambo/tasks/chem/SA_Score/fpscores.pkl.gz",
            str(fpscores_filepath),
            commit_sha="b8ea4e9",
            parent_folders_exist_ok=True,
            verbose=True,
        )


class PenalizedLogPIsolatedLogic(AbstractIsolatedFunction):
    def __init__(
        self,
        from_smiles: bool = True,
        penalized: bool = True,
    ):
        """
        TODO: document
        """
        self.from_smiles = from_smiles
        self.penalized = penalized
        _download_assets_from_lambo()

    def __call__(self, x: np.ndarray, context: dict = None):
        """
        Assuming that x is an array of strings (of shape [b,L]),
        we concatenate, translate to smiles if it's
        necessary, and then computes the penalized logP.

        If the inputs are SELFIES, it translates first to SMILES,
        and then computes the penalized logP. If the translation
        threw an error, we return NaN instead.
        """
        if x.dtype.kind not in ["U", "S"]:
            raise ValueError(
                f"We expect x to be an array of strings, but we got {x.dtype}"
            )

        molecule_strings = ["".join([x_ij for x_ij in x_i.flatten()]) for x_i in x]

        if not self.from_smiles:
            molecule_strings = translate_selfies_to_smiles(molecule_strings)

        logp_scores = []
        for molecule_string in molecule_strings:
            if molecule_string is None:
                logp_scores.append(np.nan)
            else:
                logp_scores.append(logP(molecule_string, penalized=self.penalized))

        return np.array(logp_scores).reshape(-1, 1)


if __name__ == "__main__":
    from poli.core.registry import register_isolated_function

    register_isolated_function(
        PenalizedLogPIsolatedLogic,
        name="penalized_logp_lambo__isolated",
        conda_environment_name="poli__lambo",
    )
