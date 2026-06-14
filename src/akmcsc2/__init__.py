"""akmcsc2: Active Kriging Monte Carlo Simulation with conformal certification.

Reference implementation of AK-MCS-C^2 (Jaber, Chabridon, Mougeot),
including classical U / EFF learning functions and the conformal C^2
criterion built on the J+GP and J-minmax-GP cross-conformal estimators.
"""

from .core import AKMCS, AKMCSResult, HistoryRecord
from .gp import GPRegressor, LOOGPRegressor
from .conformal import ConformalGPRegressor, JPlusGP, JMinmaxGP, make_conformal
from .learning import (
    LearningFunction,
    UFunction,
    ExpectedFeasibility,
    ConformalC2,
    IterationState,
    Selection,
    register_learning,
    make_learning,
)
from .problems import (
    ReliabilityProblem,
    four_branch,
    modified_rastrigin,
    linear_oscillator,
    get_problem,
    PROBLEMS,
)

__version__ = "0.1.0"

__all__ = [
    "AKMCS", "AKMCSResult", "HistoryRecord",
    "GPRegressor", "LOOGPRegressor",
    "ConformalGPRegressor", "JPlusGP", "JMinmaxGP", "make_conformal",
    "LearningFunction", "UFunction", "ExpectedFeasibility", "ConformalC2",
    "IterationState", "Selection", "register_learning", "make_learning",
    "ReliabilityProblem", "four_branch", "modified_rastrigin",
    "linear_oscillator", "get_problem", "PROBLEMS",
]
