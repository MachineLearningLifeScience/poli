import numpy as np

from multiprocessing import Pool, cpu_count

from poli.core.problem_setup_information import ProblemSetupInformation

from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.batch import batched


class AbstractBlackBox:
    def __init__(
        self,
        info: ProblemSetupInformation,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
    ):
        self.info = info
        self.observer = None
        self.parallelize = parallelize

        if num_workers is None:
            num_workers = cpu_count() // 2

        self.num_workers = num_workers

        if not self.info.sequences_are_aligned():
            assert (
                batch_size is None or batch_size == 1
            ), "For unaligned problems only batch size 1 is supported!"
            batch_size = 1
        self.batch_size = batch_size

    def set_observer(self, observer: AbstractObserver):
        self.observer = observer

    def __call__(self, x: np.array, context=None):
        """
        The purpose of this function is to enforce that inputs are equal across problems.
        The inputs are usually assumed to be strings, and all objective functions in our
        repository will assume that the inputs are strings. Some also allow for integers
        (i.e. token ids) to be passed as inputs.
        :param x:
        :param context:
        :return:
        """
        # We will always assume that the inputs is a 2D array.
        # The first dimension is the batch size, and the second
        # dimension should match the maximum sequence length of
        # the problem. One can opt out of this by setting L=np.inf
        # or L=None.
        if self.info.sequences_are_aligned():
            assert len(x.shape) == 2
        maximum_sequence_length = self.info.get_max_sequence_length()

        # We assert that the length matches L if the maximum sequence length
        # specified in the problem is different from np.inf or None.
        if (
            maximum_sequence_length not in [np.inf, None]
            and self.info.sequences_are_aligned()
        ):
            assert x.shape[1] == maximum_sequence_length, (
                "The length of the input is not the same as the length of the input of the problem. "
                f"(L={maximum_sequence_length}, x.shape[1]={x.shape[1]}). "
                "If you want to allow for variable-length inputs, set L=np.inf or L=None."
            )

        # Evaluate f by batches. If batch_size is None, then we evaluate
        # the whole input at once.
        batch_size = self.batch_size if self.batch_size is not None else x.shape[0]
        f_evals = []

        # We evaluate x in batches.
        for x_batch_ in batched(x, batch_size):
            # We reshape the batch to be 2D, even if the batch size is 1.
            x_batch = (
                np.concatenate(x_batch_, axis=0).reshape(len(x_batch_), -1)
                if batch_size > 1
                else np.array(x_batch_)
            )

            # We evaluate the batch in parallel if the user wants to.
            if self.parallelize:
                with Pool(self.num_workers) as pool:
                    f_batch_ = pool.starmap(
                        self._black_box, [(x.reshape(1, -1), context) for x in x_batch]
                    )
                    f_batch = np.array(f_batch_).reshape(len(x_batch_), -1)
            else:
                f_batch = self._black_box(x_batch, context)

            assert (
                len(f_batch.shape) == 2
            ), f"len(f(x).shape)={len(f_batch.shape)}, expected 2"

            assert isinstance(
                f_batch, np.ndarray
            ), f"type(f)={type(f_batch)}, not np.ndarray"

            # We pass the information to the observer, if any.
            if self.observer is not None:
                self.observer.observe(x_batch, f_batch, context)

            f_evals.append(f_batch)

        # Finally, we append the results of the batches.
        f = np.concatenate(f_evals, axis=0)
        return f

    def _black_box(self, x, context=None):
        raise NotImplementedError("abstract method")

    def terminate(self):
        pass

    def __enter__(self):
        return self

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
        super().__init__(info=f.info, batch_size=f.batch_size)

    def __call__(self, x, context=None):
        return -self.f.__call__(x, context)

    def _black_box(self, x, context=None):
        return self.f._black_box(x, context)
