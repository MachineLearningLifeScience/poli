"""This module implements the abstract black box class, from which
all objective functions should inherit.
"""

from multiprocessing import Pool, cpu_count
from warnings import warn

import numpy as np

from poli.core.black_box_information import BlackBoxInformation
from poli.core.exceptions import BudgetExhaustedException
from poli.core.util.abstract_observer import AbstractObserver
from poli.core.util.alignment import is_aligned_input
from poli.core.util.batch import batched


class AbstractBlackBox:
    """
    Abstract base class for a black box optimization problem.

    Parameters
    ----------
    batch_size : int, optional
        The batch size for evaluating the black box function. Default is None.
    parallelize : bool, optional
        Flag indicating whether to evaluate the black box function in parallel.
        Default is False.
    num_workers : int, optional
        The number of workers to use for parallel evaluation. Default is None,
        which uses half of the available CPU cores.
    evaluation_budget : int, optional
        The maximum number of evaluations allowed for the black box function.
        Default is float("inf").

    Attributes
    ----------
    observer : AbstractObserver or None
        The observer object for recording observations during evaluation.
    observer_info : object or None
        The information given by the observer after initialization.
    parallelize : bool
        Flag indicating whether to evaluate the black box function in parallel.
    num_workers : int
        The number of workers to use for parallel evaluation.
    batch_size : int or None
        The batch size for evaluating the black box function.

    Methods
    -------
    set_observer(observer)
        Set the observer object for recording observations during evaluation.
    reset_evaluation_budget()
        Reset the evaluation budget by setting the number of evaluations made
        to 0.
    __call__(x, context=None)
        Evaluate the black box function for the given input.
    _black_box(x, context=None)
        Abstract method for evaluating the black box function.
    terminate()
        Terminate the black box optimization problem.
    __enter__()
        Enter the context manager.
    __exit__(exc_type, exc_val, exc_tb)
        Exit the context manager.
    __del__()
        Destructor for the black box optimization problem.
    __neg__()
        Create a new black box with the objective function as the negative
        of the original one.
    """

    def __init__(
        self,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ):
        """
        Initialize the AbstractBlackBox object.

        Parameters
        ----------
        batch_size : int, optional
            The batch size for parallel execution, by default None.
        parallelize : bool, optional
            Flag indicating whether to parallelize the execution, by default False.
        num_workers : int, optional
            The number of workers for parallel execution, by default we use half the available CPUs.
        evaluation_budget : int, optional
            The maximum number of evaluations allowed for the black box function, by default float("inf").
        """
        self.observer = None
        self.observer_info = None
        self.parallelize = parallelize
        self.evaluation_budget = evaluation_budget
        self.num_evaluations = 0
        self.force_isolation = force_isolation

        if num_workers is None:
            num_workers = cpu_count() // 2

        self.num_workers = num_workers

        self.batch_size = batch_size

    def get_black_box_info(self) -> BlackBoxInformation:
        raise NotImplementedError(
            "Black box information must be implemented in subclasses."
        )

    @property
    def info(self) -> BlackBoxInformation:
        return self.get_black_box_info()

    def set_evaluation_budget(self, evaluation_budget: int):
        """
        Set the evaluation budget for the black box function.

        Parameters
        ----------
        evaluation_budget : int
            The maximum number of evaluations allowed for the black box function.
        """
        self.evaluation_budget = evaluation_budget

    def set_observer(self, observer: AbstractObserver):
        """
        Set the observer object for recording observations during evaluation.

        Parameters
        ----------
        observer : AbstractObserver
            The observer object.
        """
        if not isinstance(observer, AbstractObserver):
            warn(
                f"poli 🧪: Observer is not an instance of AbstractObserver: {observer} "
                "Are you sure you want to set this observer?"
            )
        self.observer = observer

    def set_observer_info(self, observer_info: object):
        """
        Set the observer information after initialization.

        Parameters
        ----------
        observer_info : object
            The information given by the observer after initialization.
        """
        self.observer_info = observer_info

    def reset_evaluation_budget(self):
        """Resets the evaluation budget by setting the number of evaluations made to 0."""
        self.num_evaluations = 0

    def __call__(self, x: np.array, context=None):
        """Calls the black box function.

        The purpose of this function is to enforce that inputs are equal across
        problems. The inputs are usually assumed to be strings, and all
        objective functions in our repository will assume that the inputs are
        strings. Some also allow for integers (i.e. token ids) to be passed as
        inputs. Some problems have continuous inputs, too.

        Parameters
        ----------
        x : np.array
            The input to the black box function.
        context : object, optional
            Additional context information for the evaluation, by default None.

        Returns
        -------
        y: np.array
            The output of the black box function.

        Raises
        ------
        AssertionError
            If the input is not a 2D array (when the sequences were supposed to be aligned).
        AssertionError
            If the length of the input does not match the maximum sequence length of the problem.
        AssertionError
            If the output is not a 2D array.
        AssertionError
            If the length of the output does not match the length of the input.
        """
        # Maximum sequence length: This can be an integer L, or np.inf, or None.
        maximum_sequence_length = self.info.get_max_sequence_length()

        # We assume that the input is either a 2D array of shape [b, L],
        # or a 1D array of shape [b,]. If the input is a 1D array, then
        # we inflate the input to be [b, 1] (allowing for variable-length
        # inputs)
        if self.info.sequences_are_aligned():
            # If the user passed an array of shape [b,],
            # then we need to make sure that the sequences
            # are aligned. Otherwise, we will raise an error.
            assert is_aligned_input(
                x, maximum_sequence_length=maximum_sequence_length
            ), (
                "The input given is not suitable for aligned problems. "
                "This may happen because the input is not an array of "
                "strings, the input is not of shape [b, L], or the input "
                "is of shape [b, ], but the lengths of the strings are "
                "not the same. "
                " Aligned problems, by definition, have as input arrays"
                " of the same length."
            )
        else:
            # At this point, we know that the sequences are not aligned.
            # Which means that the user is likely to pass an array [b,]
            # instead of an array [b, L].
            # We will reshape the input to be [b, 1] if it is not already.
            # This is necessary for batching.
            if len(x.shape) == 1:
                x = x.reshape(-1, 1)

        # We assert that the length matches L if the maximum sequence length
        # specified in the problem is different from np.inf or None.
        if (
            maximum_sequence_length not in [np.inf, None]
            and self.info.sequences_are_aligned()
        ):
            if len(x.shape) == 1:
                assert len(set([len(s) for s in x])) == 1, (
                    "The sequences are not aligned. "
                    "If you want to allow for variable-length inputs, set "
                    "the maximum length of the problem to be L=np.inf or L=None. "
                    "(and the aligned flag to False)."
                )
                x = x.reshape(-1, 1)
            else:
                assert x.shape[1] == maximum_sequence_length, (
                    "The length of the input is not the same as the length of the input of the problem. "
                    f"(L={maximum_sequence_length}, x.shape[1]={x.shape[1]}). "
                    "If you want to allow for variable-length inputs, set L=np.inf or L=None."
                )

        # Evaluate f by batches. If batch_size is None, then we evaluate
        # the whole input at once.
        if self.batch_size is None:
            batch_size = x.shape[0]
        else:
            batch_size = self.batch_size
        f_evals = []

        # Check whether we have enough budget to evaluate the black box function
        # in the current batch.
        if self.num_evaluations + batch_size > self.evaluation_budget:
            raise BudgetExhaustedException(
                f"Exhausted the evaluation budget of {self.evaluation_budget} evaluations."
                f" (tried to evaluate {batch_size}, but we have already"
                f" evaluated {self.num_evaluations}/{self.evaluation_budget})."
            )

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
            else:  # iterative treatment
                f_batch = self._black_box(x_batch, context)

            assert (
                len(f_batch.shape) == 2
            ), f"len(f(x).shape)={len(f_batch.shape)}, expected 2"

            assert isinstance(
                f_batch, np.ndarray
            ), f"type(f)={type(f_batch)}, not np.ndarray"

            assert (
                x_batch.shape[0] == f_batch.shape[0]
            ), f"Inconsistent evaluation axis=0 x={x_batch.shape} != black_box y={f_batch.shape}"

            # We pass the information to the observer, if any.
            if self.observer is not None:
                # observer logic s.t. observations are individual - later aggregate w.r.t batch_size
                if x_batch.shape[0] > 1:
                    for i in range(x_batch.shape[0]):
                        _x = np.atleast_2d(x_batch[i])
                        _y = np.atleast_2d(f_batch[i])
                        self.observer.observe(_x, _y, context)
                else:
                    self.observer.observe(x_batch, f_batch, context)

            f_evals.append(f_batch)

            # We update the number of evaluations.
            self.num_evaluations += x_batch.shape[0]

        # Finally, we append the results of the batches.
        f = np.concatenate(f_evals, axis=0)
        return f

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
        """
        Abstract method for evaluating the black box function.

        Parameters
        ----------
        x : np.ndarray
            The input values to evaluate the black box function.
        context : any, optional
            Additional context information for the evaluation. Defaults to None.

        Returns
        -------
        y: np.ndarray
            The output of the black box function.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the derived class.
        """
        raise NotImplementedError("abstract method")

    def terminate(self) -> None:
        """
        Terminate the black box optimization problem.
        """
        if hasattr(self, "inner_function"):
            self.inner_function.terminate()
        # if self.observer is not None:
        #     # NOTE: terminating a problem should gracefully end the observer process -> write the last state.
        #     self.observer.finish()
        return

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

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(L={self.info.max_sequence_length}, num_evaluations={self.num_evaluations})"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(L={self.info.max_sequence_length}, batch_size={self.batch_size}, parallelize={self.parallelize}, num_workers={self.num_workers}, evaluation_budget={self.evaluation_budget})>"


class NegativeBlackBox(AbstractBlackBox):
    """A wrapper for a black box that negates the objective function.

    If you construct a black-box function f for maximizing, then -f is
    a black-box function for minimizing. This class is a wrapper for
    implementing the latter.

    The only difference is that the __call__ method returns -f(x) instead
    of f(x). The _black_box method is the same as the original black box.
    """

    def __init__(self, f: AbstractBlackBox):
        self.f = f
        super().__init__(
            batch_size=f.batch_size,
            parallelize=f.parallelize,
            num_workers=f.num_workers,
            evaluation_budget=f.evaluation_budget,
        )

    def __call__(self, x, context=None):
        return -self.f.__call__(x, context)

    def _black_box(self, x, context=None):
        return self.f._black_box(x, context)

    def __str__(self) -> str:
        return f"NegativeBlackBox({self.f})"

    def __repr__(self) -> str:
        return f"<NegativeBlackBox({self.f})>"
