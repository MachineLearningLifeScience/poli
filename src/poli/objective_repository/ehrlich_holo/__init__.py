"""A closed-form black box simulating epistatic effects [1].

References
----------
[1] Stanton, S., Alberstein, R., Frey, N., Watkins, A., & Cho, K. (2024).
    Closed-Form Test Functions for Biophysical Sequence Optimization Algorithms.
    arXiv preprint arXiv:2407.00236. https://arxiv.org/abs/2407.00236
"""

from .register import EhrlichHoloBlackBox, EhrlichHoloProblemFactory

__all__ = ["EhrlichHoloBlackBox", "EhrlichHoloProblemFactory"]
