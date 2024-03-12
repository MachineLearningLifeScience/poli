"""Custom exceptions inside the poli package."""


class PoliException(Exception):
    """Base class for exceptions in the poli package."""

    pass


class BudgetExhaustedException(PoliException):
    """Exception raised when the budget is exhausted."""

    pass


class FoldXNotFoundException(PoliException):
    """Exception raised when FoldX wasn't found in ~/foldx/foldx."""

    pass
