"""
This module contains coding for training a Gaussian Process Regression
model on the Sarkisyan (2016) data set.
Plus some utility constants.
"""

import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np

from poli.objective_repository.gfp_cbas import BLOSUM


class SequenceGP(object):
    def __init__(
        self,
        load: bool = False,
        X_train: np.ndarray = None,
        y_train: np.ndarray = None,
        length_scale: float = 1,
        homo_noise: float = 0.1,
        load_path_prefix: Path = Path(__file__).parent
        / "assets"
        / "models"
        / "gp"
        / "gfp_gp",
        k_beta: float = 0.1,
        c: float = 1,
        d: float = 2,
    ):
        if load:
            self.load(path=load_path_prefix)
        else:
            assert X_train is not None and y_train is not None
            self.X_ = np.copy(X_train)
            self.y_ = np.copy(y_train).reshape((y_train.shape[0], 1))
            self.N_ = self.X_.shape[0]
            self.params_ = np.array([homo_noise, k_beta, c, d])
            self.K_ = None
            self.Kinv_ = None

    def _kernel(self, Xi: np.ndarray, Xj: np.ndarray) -> float:
        """
        BLOSUM based product kernel of discrete input sequences.
        Hyperparameters are applied
        """
        beta = self.params_[1]
        c = self.params_[2]
        d = self.params_[3]
        kij = np.prod(BLOSUM[Xi, Xj] ** beta)
        kii = np.prod(BLOSUM[Xi, Xi] ** beta)
        kjj = np.prod(BLOSUM[Xj, Xj] ** beta)
        k = kij / (np.sqrt(kii * kjj))
        k = np.exp(c * k)
        return k

    def _fill_K(self, print_every: int = 100) -> None:
        self.K_ = np.zeros((self.N_, self.N_))
        total = self.N_ * (self.N_ + 1) / 2
        m = 0
        homo_noise = self.params_[0]
        for i in range(self.N_):
            for j in range(i, self.N_):
                kij = self._kernel(self.X_[i], self.X_[j])
                if i == j:
                    kij += homo_noise
                self.K_[i, j] = kij
                self.K_[j, i] = kij

                m += 1
                if m % print_every == 0:
                    print("Number of K elements filled: %i / %i" % (m, total))

    def _invert_K(self):
        print("Inverting K...")
        self.Kinv_ = np.linalg.inv(self.K_)
        print("Done inverting K.")

    def build(self, print_every=100):
        self._fill_K(print_every=print_every)
        self._invert_K()

    def predict(
        self, Xstar: np.ndarray, print_every: int = None, predict_variance: bool = False
    ) -> np.ndarray:
        """
        GP posterior predictive, compute kernel values across all new Xstar and existing observations X_ .
        """
        M = len(Xstar)
        Kstar = np.zeros((M, self.N_))
        total = M * self.N_
        m = 0
        for i in range(M):
            for j in range(self.N_):
                kij = self._kernel(Xstar[i], self.X_[j])
                Kstar[i, j] = kij
                m += 1
                if print_every is not None:
                    if m % print_every == 0:
                        print("Number of Kstar elements filled: %i / %i" % (m, total))
        mu_star = np.matmul(Kstar, np.matmul(self.Kinv_, self.y_))
        return mu_star

    def save(self, path: Path):
        """
        Persist numpy arrays from object properties.
        """
        np.save(path.parent / (path.name + "X.npy"), self.X_)
        np.save(path.parent / (path.name + "y.npy"), self.y_)
        np.save(path.parent / (path.name + "K.npy"), self.K_)
        np.save(path.parent / (path.name + "Kinv.npy"), self.Kinv_)
        np.save(path.parent / (path.name + "params.npy"), self.params_)

    def load(self, path: Path):
        """
        Load persisted files. Sets object properties.
        """
        self.X_ = np.load(path.parent / (path.name + "X.npy"))
        self.y_ = np.load(path.parent / (path.name + "y.npy"))
        self.K_ = np.load(path.parent / (path.name + "K.npy"))
        self.Kinv_ = np.load(path.parent / (path.name + "Kinv.npy"))
        self.params_ = np.load(path.parent / (path.name + "params.npy"))
        self.N_ = self.X_.shape[0]
