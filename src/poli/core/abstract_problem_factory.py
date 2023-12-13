"""This module implements the abstract problem factory."""
from typing import Tuple

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.problem_setup_information import ProblemSetupInformation


class MetaProblemFactory(type):
    """
    Metaclass for the AbstractProblemFactory class.
    (which allows us to override the __repr__ and __str__ methods)
    """

    def __repr__(cls) -> str:
        try:
            problem_info = cls().get_setup_information()
        except NotImplementedError:
            return f"<{cls.__name__}()>"

        return f"<{cls.__name__}(L={problem_info.max_sequence_length})>"

    def __str__(cls) -> str:
        return f"{cls.__name__}"


class AbstractProblemFactory(metaclass=MetaProblemFactory):
    """
    Abstract base class for problem factories.

    This class defines the interface for creating problem instances in the POLI framework.

    Attributes:
    -----------
        None

    Methods:
    --------
    get_setup_information:
        Returns the setup information for the problem.
    create:
        Creates a problem instance with the specified parameters.
    """

    def get_setup_information(self) -> ProblemSetupInformation:
        """
        Returns the setup information for the problem.

        Returns:
        --------
        problem_info: ProblemSetupInformation
            The setup information for the problem.

        Raises:
        -------
        NotImplementedError:
            This method is abstract and must be implemented by subclasses.
        """
        raise NotImplementedError("abstract method")

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
    ) -> Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
        """
        Returns a blackbox function and initial observations.

        Parameters:
        -----------
        seed:  int
            The seed for random number generation. Default is None.
        batch_size:  int
            The batch size for parallel evaluation. Default is None.
        parallelize : bool
            Flag indicating whether to parallelize the evaluation. Default is False.
        num_workers:  int
            The number of workers for parallel evaluation. Default is None.
        evaluation_budget:  int, optional
            The maximum number of function evaluations. Default is infinity.

        Returns:
        --------
            results: Tuple[AbstractBlackBox, np.ndarray, np.ndarray]:
                A tuple containing the blackbox function, initial observations for
                input variables, and initial observations for output variables.

        Raises:
        -------
            NotImplementedError: This method is abstract and must be implemented by subclasses.

        """
        raise NotImplementedError("abstract method")
