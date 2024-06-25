"""Default observer that just does nothing."""

from poli.core.util.abstract_observer import AbstractObserver


class DefaultObserver(AbstractObserver):
    """
    This observer serves as a place-holder if the user does not want to use observers.
    """

    def observe(self, *args, **kwargs) -> None:
        pass

    def initialize_observer(self, *args, **kwargs) -> object:
        return None
