from poli.core.util.abstract_observer import AbstractObserver


class BlackBox:
    def __init__(self, L: int):
        """
        :param L: length of the inputs (NOT 1hot encoded)
        """
        self.L = L
        self.observer = None

    def set_observer(self, observer: AbstractObserver):
        self.observer = observer

    def __call__(self, x):
        """
        The purpose of this function is to enforce that inputs are equal across problems.
        To avoid errors inputs must be of shape 1xL. (That is, NOT 1hot encoded but explicitly using the alphabet.)
        :param x:
        :type x:
        :return:
        :rtype:
        """
        assert(len(x.shape) == 2)
        assert(x.shape[0] == 1)
        assert(x.shape[1] == self.L)
        f = self._black_box(x)
        assert(len(f.shape) == 2)
        assert(f.shape[0] == 1)
        assert(f.shape[1] == 1)
        if self.observer is not None:
            self.observer.observe(x, f)
        return f

    def _black_box(self, x):
        raise NotImplementedError("abstract method")
