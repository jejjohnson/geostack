"""krigax: exact Gaussian-process regression on lineax operators.

References:
    Rasmussen, C. E. and Williams, C. K. I. (2006). Gaussian Processes for
    Machine Learning. MIT Press. Algorithm 2.1.
"""

from krigax._kernels import RBF, Matern32
from krigax._model import GaussianProcess, fit
from krigax._predict import log_marginal_likelihood, predict
from krigax._solvers import CGSolver, CholeskySolver

__all__ = [
    "RBF",
    "CGSolver",
    "CholeskySolver",
    "GaussianProcess",
    "Matern32",
    "fit",
    "log_marginal_likelihood",
    "predict",
]
