import numpy as np
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.abstract_observer import AbstractObserver


class AbstractBlackBox:
    def __init__(self, info: ProblemSetupInformation):
        self.info = info
        self.observer = None

    def set_observer(self, observer: AbstractObserver):
        self.observer = observer

    def __call__(self, x: np.array, context=None):
        """
        The purpose of this function is to enforce that inputs are equal across problems.
        To avoid errors, the inputs are strings.
        :param x:
        :param context:
        :return:
        """
        x_ = x[:1, ...]
        # TODO: activate assertion below for discrete problems
        assert(all([c in self.info.get_alphabet() for c in x_[0]]))
        f_ = self._black_box(x_, context)
        assert (isinstance(f_, np.ndarray))
        assert (len(f_.shape) == 2)
        assert (f_.shape[0] == 1)
        # TODO: consider notifying observers rather in batches
        if self.observer is not None:
            self.observer.observe(x_, f_, context)
        # assert(f_.shape[1] == 1)  # allow multi-tasking
        f = np.zeros([len(x), *f_.shape[1:]])
        f[:1, ...] = f_
        for i in range(1, x.shape[0]):
            x_ = x[i:i+1, ...]
            assert (all([c in self.info.get_alphabet() for c in x_[0]]))
            f_ = self._black_box(x_, context)
            f[i:i+1, ...] = f_
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
