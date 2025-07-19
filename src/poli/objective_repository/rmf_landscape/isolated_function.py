from __future__ import annotations

import logging

import numpy as np
from scipy.stats import genpareto  # type: ignore[reportMissingImports]

from poli.core.abstract_isolated_function import AbstractIsolatedFunction
from poli.core.util.proteins.defaults import AMINO_ACIDS, ENCODING


class RMFIsolatedLogic(AbstractIsolatedFunction):
    """
    RMF internal logic.

    Parameters
    ----------
    wildtype : list[str]
        String sequence of the reference, default: None.
    c : float, optional

    alphabet : list[str]
        Alphabet for the problem, by default AA list provided from poli.core.util.proteins.defaults
    stochasticity: str, optional
    Methods
    -------
    _black_box(x, context=None)
        Main black box method to compute the fitness value of x relative to the WT.

    Raises
    ------
    AssertionError
        If no wildtype sequence is provided.
    """

    def __init__(
        self,
        wildtype: list[str],
        wt_val: float = 0.0,
        c: float | None = None,
        kappa: float = 0.1,
        alphabet: list[str] | None = None,
        seed: int | None = 0,
    ) -> None:
        """
        Initialize the RMFBlackBox object.
        """
        assert wildtype is not None, (
            "Missing reference input sequence. "
            "Did you forget to pass it to the create of the black box?"
        )
        if not isinstance(wildtype, np.ndarray):
            wildtype_ = np.array(list(wildtype))
        else:
            wildtype_ = wildtype
        self.wildtype = wildtype_
        self.seed = seed
        if alphabet is None:
            logging.info("using default alphabet AAs.")
            alphabet = AMINO_ACIDS
        assert all(
            [aa in ENCODING.keys() for aa in wildtype_]
        ), "Input wildtype elements not in encoding alphabet."
        self.wt_int = np.array([ENCODING.get(aa) for aa in wildtype_])
        if c is None:
            c = 1 / (len(alphabet) - 1)
        else:
            c = c
        assert c >= 0, "Invalid c : c > 0 required!"
        logging.info(f"setting c={c}")
        # if c == 0 : uncorrelated HoC landscape (?)
        self.c = c
        self.kappa = kappa
        self.f_0 = (
            wt_val  # in case of standardized observations (around WT) assume w.l.o.g.
        )
        self.alphabet = alphabet
        eta_var = genpareto.stats(c, moments="v")
        self.theta = c / np.sqrt(eta_var)
        self.rng = np.random.default_rng(seed)
        logging.info(f"landscape theta={self.theta}")
        super().__init__()

    @staticmethod
    def f(
        f0: float,
        sigma: np.ndarray,
        sigma_star: np.ndarray,
        c: float,
        kappa: float,
        rand_state,
    ) -> float:
        # from [1] (2) additive term via Hamming distance and constant
        # hamm_dist = hamming(sigma.flatten(), sigma_star.flatten()) # NOTE scipy HD is normalized, DON't USE
        hamm_dist = np.sum(sigma != sigma_star)
        # from [2] nonadd. term is single small value accroding to RV, we use [1]gen.Pareto RV instead of Gaussian
        eta = genpareto.rvs(kappa, size=1, random_state=rand_state)
        # NOTE [1] describes eta as 2^L i.i.d. RV vector, which does not yield a single function value
        f_p = f0 + -c * hamm_dist
        f_val = f_p + eta
        return f_val

    def __call__(self, x: np.ndarray, context=None) -> np.ndarray:
        values = []
        for sequence in x:
            L = len(sequence)
            assert L == self.wildtype.shape[-1], "Inconsistent length: undefined."
            x_int = np.array([ENCODING.get(aa) for aa in sequence])
            val = self.f(
                f0=self.f_0,
                sigma=x_int,
                sigma_star=self.wt_int,
                c=self.c,
                kappa=self.kappa,
                rand_state=self.rng,
            )
            values.append(val)
        return np.array(values).reshape(-1, 1)


if __name__ == "__main__":
    from poli.core.registry import register_isolated_function

    register_isolated_function(
        RMFIsolatedLogic,
        name="rmf_landscape__isolated",
        conda_environment_name="poli__rmf",
        force=True,
    )
