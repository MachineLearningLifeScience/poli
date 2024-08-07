"""This module implements the abstract problem factory."""

from poli.core.problem import Problem


class MetaProblemFactory(type):
    """
    Metaclass for the AbstractProblemFactory class.
    (which allows us to override the __repr__ and __str__ methods)
    """

    def __repr__(cls) -> str:
        return f"<{cls.__name__}()>"

    def __str__(cls) -> str:
        return f"{cls.__name__}"


class AbstractProblemFactory(metaclass=MetaProblemFactory):
    """
    Abstract base class for problem factories.

    This class defines the interface for creating problem instances in poli.

    Methods
    -------
    create:
        Creates a problem instance with the specified parameters.
    """

    def create(
        self,
        seed: int = None,
        batch_size: int = None,
        parallelize: bool = False,
        num_workers: int = None,
        evaluation_budget: int = float("inf"),
        force_isolation: bool = False,
    ) -> Problem:
        """
        Returns a blackbox function and initial observations.

        Parameters
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
        force_isolation: bool, optional
            Whether to force the isolation of the black box. Default
            is False.
        Returns
        --------
        problem: AbstractProblem
            A problem class containing, among other things, the black box,
            initial values x0 and y0, and evaluation budget.

        Raises
        -------
            NotImplementedError: This method is abstract and must be implemented by subclasses.

        """
        raise NotImplementedError(
            "Abstract method create() must be implemented by subclasses."
        )
