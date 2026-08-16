"""Uplift modeling & incremental impact simulator."""

from .bayesian import (
    BayesianBootstrapUplift,
    coverage_summary,
    decision_rule,
    group_coverage,
    inflate_posterior,
)
from .config import DATASETS, CRITEO, HILLSTROM, Economics, get_spec
from .data import average_treatment_effect, prepare, randomization_check
from .experiment import (
    design_decay_monitor,
    design_policy_test,
    design_profit_test,
    mde_for_sample_size,
    sample_size_two_proportions,
)
from .evaluate import auuc, decile_table, evaluate, qini_coefficient, qini_curve
from .learners import SLearner, TLearner, XLearner, build_learners
from .monitoring import UpliftDriftMonitor, realized_effect
from .pipeline import UpliftBundle, build_bundle, select_best, train_all
from .segments import classify, summarize
from .simulate import compare_policies, headline, optimal_depth, simulate_policy

__version__ = "1.0.0"

__all__ = [
    "DATASETS", "CRITEO", "HILLSTROM", "Economics", "get_spec",
    "prepare", "randomization_check", "average_treatment_effect",
    "qini_curve", "qini_coefficient", "auuc", "decile_table", "evaluate",
    "SLearner", "TLearner", "XLearner", "build_learners",
    "classify", "summarize",
    "simulate_policy", "compare_policies", "optimal_depth", "headline",
    "UpliftDriftMonitor", "realized_effect",
    "UpliftBundle", "train_all", "select_best", "build_bundle",
    "BayesianBootstrapUplift", "group_coverage", "coverage_summary",
    "decision_rule", "inflate_posterior",
    "sample_size_two_proportions", "mde_for_sample_size",
    "design_policy_test", "design_profit_test", "design_decay_monitor",
]
