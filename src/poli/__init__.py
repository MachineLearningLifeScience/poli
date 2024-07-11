"""poli, a library for discrete black-box objective functions."""

__author__ = "Simon Bartels & Miguel González-Duque (MLLS)"
from .core.util.isolation.instancing import instance_function_as_isolated_process

# from .core import get_problems
from .objective_factory import create
from .objective_repository import get_problems
