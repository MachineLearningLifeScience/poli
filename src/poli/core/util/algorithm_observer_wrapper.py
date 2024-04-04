"""A wrapper around observers that prohibits the use of observer.observe"""

from poli.core.util.abstract_observer import AbstractObserver


class AlgorithmObserverWrapper:
    def __init__(self, observer: AbstractObserver):
        self._observer = observer

    def log(self, algorithm: dict):
        self._observer.log(algorithm)
