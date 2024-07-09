import logging
from typing import List, Optional

import numpy as np
from scipy.spatial.distance import hamming
from scipy.stats import genpareto

from poli.core.abstract_isolated_function import AbstractIsolatedFunction
from poli.core.util.proteins.defaults import AMINO_ACIDS, ENCODING


class RMFIsolatedLogic(AbstractIsolatedFunction):
    """
    RMF internal logic.
    
    Parameters
    ----------
    wildtype : List[str]
        String sequence of the reference, default: None.
    c : float, optional

    alphabet : List[str]
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
        wildtype: List[str], 
        wt_val: Optional[float]=0., 
        c: Optional[float]=None, 
        kappa: Optional[float]=0.1, 
        alphabet: Optional[List[str]]=None, 
        seed: Optional[int]=0
    ) -> None:
        """
        Initialize the RMFBlackBox object.
        """
        assert wildtype is not None, (
            "Missing reference input sequence. "
            "Did you forget to pass it to the create of the black box?"
        )
        oracle_name = "RMF"
        if not isinstance(wildtype, np.ndarray):
            wildtype = np.array(list(wildtype))
        self.wildtype = wildtype
        self.seed = seed
        if alphabet is None:
            logging.info("using default alphabet AAs.")
            alphabet = AMINO_ACIDS
        assert all([aa in ENCODING.keys() for aa in wildtype]) , "Input wildtype elements not in encoding alphabet."
        wt_int = [ENCODING.get(aa) for aa in wildtype]
        wt_oh = np.zeros((len(wildtype), len(AMINO_ACIDS)))
        wt_oh[np.arange(len(wildtype)), wt_int] = 1
        self.wt_oh = wt_oh
        if c is None:
            c = 1 / (len(alphabet)-1)
        else:
            c = c
        assert c >= 0, "Invalid c : c > 0 required!"
        logging.info(f"setting c={c}")
        # if c == 0 : uncorrelated HoC landscape (?)
        self.c = c
        self.kappa = kappa 
        self.f_0 = wt_val # in case of standardized observations (around WT) assume w.l.o.g.
        self.alphabet = alphabet
        eta_var = genpareto.stats(c, moments='v')
        self.theta = c / np.sqrt(eta_var)
        logging.info(f"landscape theta={self.theta}")
        super().__init__()

    @staticmethod
    def f(f0: float, sigma: np.ndarray, sigma_star: np.ndarray, c: float, kappa: float) -> float:
        L = len(sigma)
        # from [1] (2) additive term via Hamming distance and constant
        print(sigma.shape)
        print(sigma_star.shape)
        hamm_dist = hamming(sigma.flatten(), sigma_star.flatten())
        print(hamm_dist)
        # from [2] nonadd. term is single small value accroding to RV, we use [1] RV instead of Gaussian
        eta = genpareto.rvs(kappa, size=1)
        # NOTE [1] describes eta as 2^L i.i.d. RV vector, which does not yield a single function value
        f_p = f0 + np.sum(-c * hamm_dist) / L
        f_val = f_p + eta
        # print(f_val)
        return f_val
    
    def __call__(self, x: np.ndarray, context=None) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(list(x))
        assert x.shape[-1] == self.wildtype.shape[-1], "Inconsistent length: undefined."
        x_int = np.array([[ENCODING.get(aa) for aa in _x] for _x in x])
        x_oh = np.zeros((x_int.shape[-1], len(AMINO_ACIDS)))
        x_oh[np.arange(x_int.shape[-1]), x_int] = 1
        val = self.f(
            f0 = self.f_0,
            sigma=x_oh, 
            sigma_star=self.wt_oh,
            c=self.c,
            kappa=self.kappa,
        )
        return np.array(val).reshape(-1, 1)
    

if __name__ == "__main__":
    from poli.core.registry import register_isolated_function

    register_isolated_function(
        RMFIsolatedLogic,
        name="rmf__isolated",
        conda_environment_name="poli__rmf",
        force=True,
    ) 