"""Abstract class for observers in poli."""

import numpy as np

from poli.core.black_box_information import BlackBoxInformation


class AbstractObserver:
    """
    Abstract base class for observers in the poli library.

    Observers are used to monitor the progress of optimization algorithms
    by observing the values of the objective function and the decision variables
    at each iteration.

    IMPORTANT: The __init__ function of observer classes MUST NOT require arguments.

    Methods
    -------
        observe(x: np.ndarray, y: np.ndarray, context=None) -> None:
            Observes the values of the objective function and the decision variables
            at each iteration of the optimization algorithm. If the observer is
            set in the creation of an objective function, this method will be
            called every time the objective function is called.

        initialize_observer(problem_setup_info: ProblemSetupInformation, caller_info: object, seed: int) -> object:
            Initializes the observer with the necessary information to monitor
            the optimization process.

        finish() -> None:
            Performs any necessary cleanup or finalization steps after the
            optimization process is complete.
    """

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        """
        Observe the given data points.

        Parameters
        ----------
        x: np.ndarray
            The input data points.
        y: np.ndarray
            The output data points.
        context: object
            Additional context for the observation.

        Raises
        -------
        NotImplementedError:
            This method is meant to be overridden by subclasses.
        """
        raise NotImplementedError("abstract method")

    def initialize_observer(
        self,
        problem_setup_info: BlackBoxInformation,
        caller_info: object,
        seed: int,
    ) -> object:
        """
        Initialize the observer.

        Parameters
        ----------
        problem_setup_info : BlackBoxInformation
            The information about the problem setup.
        caller_info : object
            The information that is passed to the observer at init time.
        seed : int
            The seed value for random number generation.

        Returns
        -------
        observer: AbstractObserver (self)
            The observer object. (This way, one can chain the `initialize_observer` method.)

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the derived class.

        """
        raise NotImplementedError("abstract method")

    def log(self, algorithm_info: dict):
        pass

    def finish(self) -> None:
        """Finish the observer."""
        pass

    def __del__(self):
        try:
            self.finish()
        except Exception:
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
