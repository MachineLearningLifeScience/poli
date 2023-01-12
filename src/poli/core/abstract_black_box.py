import numpy as np

from poli.core.util.abstract_observer import AbstractObserver


class AbstractBlackBox:
    def __init__(self, L: int):
        """
        :param L: length of the inputs (NOT 1hot encoded)
        """
        self.L = L
        self.sequences_aligned = True  # TODO: make this an option
        self.observer = None

    def set_observer(self, observer: AbstractObserver):
        self.observer = observer

    def __call__(self, x, context=None):
        """
        The purpose of this function is to enforce that inputs are equal across problems.
        To avoid errors inputs must be of shape 1xL. (That is, NOT 1hot encoded but explicitly using the alphabet.)
        :param x:
        :param context:
        :return:
        """
        assert(len(x.shape) == 2)
        #assert(x.shape[0] == 1)
        assert(x.shape[1] == self.L or not self.sequences_aligned)
        f = np.zeros([x.shape[0], 1])
        for i in range(x.shape[0]):
            x_ = x[i:i+1, :]
            f_ = self._black_box(x_, context)
            f[i] = f_
            assert(len(f_.shape) == 2)
            assert(f_.shape[0] == 1)
            assert(f_.shape[1] == 1)
            assert(isinstance(f, np.ndarray))
            if self.observer is not None:
                self.observer.observe(x_, f_, context)
        return f

    def _black_box(self, x, context=None):
        raise NotImplementedError("abstract method")
