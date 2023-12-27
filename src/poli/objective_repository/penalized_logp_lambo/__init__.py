"""The exact same computation of LogP used by LamBO.

In [1], the authors optimize jointly for the Quantitative Estimate of
Druglikeness (QED) and a penalized version of log-solubility (logP).
For fair comparisons, we implement the exact same computation of logP.

References
----------
[1] “Accelerating Bayesian Optimization for Biological Sequence Design with Denoising Autoencoders.”
Stanton, Samuel, Wesley Maddox, Nate Gruver, Phillip Maffettone,
Emily Delaney, Peyton Greenside, and Andrew Gordon Wilson.
arXiv, July 12, 2022. http://arxiv.org/abs/2203.12742.
"""
