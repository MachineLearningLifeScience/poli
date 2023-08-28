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
        assert len(x.shape) == 2
        # assert(x.shape[0] == 1)
        # self.L == np.inf is a way to say that the
        # length of the input is not fixed.
        if self.L is not np.inf:
            assert x.shape[1] == self.L or not self.sequences_aligned, (
                "The length of the input is not the same as the length of the input of the problem. "
                f"(L={self.L}, x.shape[1]={x.shape[1]}). "
                "If you want to allow for variable length inputs, set L=np.inf."
            )
        # TODO: what happens with multi-objective?
        # In some cases, we might be interested in functions
        # that output more than one value.
        # TODO: What happens with batched inputs?
        # Why do we want to evaluate the objective
        # function one at a time?
        f = np.zeros([x.shape[0], 1])
        for i in range(x.shape[0]):
            x_ = x[i : i + 1, :]
            f_ = self._black_box(x_, context)
            f[i] = f_
            assert len(f_.shape) == 2, f"len(f_.shape)={len(f_.shape)}, expected 2"
            assert f_.shape[0] == 1, f"f_.shape[0]={f_.shape[0]}, expected 1"
            assert f_.shape[1] == 1, f"f_.shape[1]={f_.shape[1]}, expected 1"
            assert isinstance(f, np.ndarray), f"type(f)={type(f)}, not np.ndarray"
            if self.observer is not None:
                self.observer.observe(x_, f_, context)
        return f

    def _black_box(self, x, context=None):
        raise NotImplementedError("abstract method")

    def terminate(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminate()

    def __del__(self):
        self.terminate()

    def __neg__(self):
        """
        Creates a new black box, where the objective function
        is the negative of the original one.
        """
        negative_black_box = NegativeBlackBox(self)
        return negative_black_box


class NegativeBlackBox(AbstractBlackBox):
    def __init__(self, f: AbstractBlackBox):
        self.f = f
        super().__init__(f.L)

    def __call__(self, x, context=None):
        return -self.f.__call__(x, context)

    def _black_box(self, x, context=None):
        return self.f._black_box(x, context)
