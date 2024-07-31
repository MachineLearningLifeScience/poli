"""
Implements a multi_objective version of black box functions,
by this we mean simply concatenating the results of evaluating individual
objective functions.
"""

from typing import List

import numpy as np

from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.black_box_information import BlackBoxInformation


class MultiObjectiveBlackBox(AbstractBlackBox):
    """
    This class implements a multi-objective black box function
    by concatenating the results of evaluating individual
    objective functions.

    Parameters
    -----------
    batch_size : int, optional
        The batch size for evaluating the black box function. Defaults to None.
    objective_functions : List[AbstractBlackBox], required
        The list of objective functions to be evaluated. Defaults to None.

    Attributes
    ----------
    objective_functions : List[AbstractBlackBox]
        The list of objective functions to be evaluated.

    Methods
    -------
    _black_box(x, context=None)
        Evaluate the black box function for the given input.

    Raises
    ------
    ValueError
        If objective_functions is not provided as a list of AbstractBlackBox
        instances or inherited classes.
    """

    # TODO: Should we evaluate in parallel?
    # TODO: At some point, we should consider unions of ProblemSetupInformations.
    #       That way, we could simply specify the list of black box functions, and
    #       then "join" their problem setups.

    def __init__(
        self,
        objective_functions: List[AbstractBlackBox],
        batch_size: int = None,
    ) -> None:
        """
        Initialize the MultiObjectiveBlackBox class.

        Parameters
        -----------
        objective_functions : List[AbstractBlackBox]
            The list of objective functions.
        batch_size : int, optional
            The batch size. Defaults to None.

        Raises
        -------
        ValueError:
            If objective_functions is not provided as a list of AbstractBlackBox instances or inherited classes.
        """
        if objective_functions is None:
            raise ValueError(
                "objective_functions must be provided as a list of AbstractBlackBox instances or inherited classes."
            )

        super().__init__(
            batch_size=batch_size,
        )

        self.objective_functions = objective_functions

    def _black_box(self, x: np.ndarray, context=None) -> np.ndarray:
        """
        Evaluate the black box function for a given input.

        Parameters
        -----------
        x : np.ndarray
            The input values to evaluate the black box function.
        context : any, optional
            Additional context information for the evaluation.

        Returns
        --------
            array-like: The concatenated results of evaluating the objective functions on the input.
        """
        # TODO: Decide on whether we want to evaluate on parallel or not.
        res = []
        for obj_function in self.objective_functions:
            res.append(obj_function(x, context))

        return np.concatenate(res, axis=1)

    def __str__(self) -> str:
        return f"MultiObjectiveBlackBox(black_boxes={self.objective_functions})"

    def __repr__(self) -> str:
        return f"<MultiObjectiveBlackBox(black_boxes={self.objective_functions}, batch_size={self.batch_size})>"

    @property
    def info(self) -> BlackBoxInformation:
        """
        Return the problem setup information for the multi-objective black box.

        Returns
        -------
        BlackBoxInformation:
            Information about the first objective function.
        """
        # TODO: what should this return, actually?
        # I'd say that we expect all objective functions to be able
        # to take the same input.
        return self.objective_functions[0].info
