"""poli, a library for discrete black-box objective functions."""

__author__ = "Simon Bartels & Miguel González-Duque (MLLS)"
from .core import get_problems
from .objective_factory import create_problem
from .core.util.isolation.instancing import instance_black_box_as_isolated_process
