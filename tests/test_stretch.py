"""Tests for the uncertainty and experiment-design layers.

Same principle as the main suite: check things that must hold by construction,
and things whose failure would be invisible. A posterior that silently collapses
to zero width still produces confident-looking numbers, and a sample-size
formula that is wrong by a factor of four still returns a plausible integer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uplift.bayesian import (
    BayesianBootstrapUplift,
    coverage_summary,
    decision_rule,
    group_coverage,
    inflate_posterior,
)
from uplift.experiment import (
    design_decay_monitor,
    design_policy_test,
    design_profit_test,
    duration_table,
    mde_for_sample_size,
    power_curve,
    sample_size_two_proportions,
)
from uplift.learners import SLearner, TLearner


@pytest.fixture
def synthetic_positive():
    """Randomized experiment where every customer's effect is positive.

    Separate from ``synthetic`` because several invariants below only hold
    without sleeping dogs — a distinction that is easy to state loosely and
    get wrong.
    """
    rng = np.random.default_rng(5)
    n = 8_000
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=[f"x{i}" for i in range(3)])
    t = rng.binomial(1, 0.5, n)
    true_tau = 0.02 + 0.10 * (X["x0"] - X["x0"].min()) / (X["x0"].max() - X["x0"].min())
    p = np.clip(0.25 + t * true_tau, 0.01, 0.99)
    y = rng.binomial(1, p)
    return X, t, y, true_tau.to_numpy()


@pytest.fixture
def synthetic():
    """Randomized experiment with a known, feature-driven treatment effect.

    Roughly half the population has a *negative* effect, so this fixture
    contains genuine sleeping dogs.
    """
    rng = np.random.default_rng(0)
    n = 8_000
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=[f"x{i}" for i in range(3)])
    t = rng.binomial(1, 0.5, n)
    true_tau = 0.15 * X["x0"]
    p = np.clip(0.25 + t * true_tau, 0.01, 0.99)
    y = rng.binomial(1, p)
    return X, t, y, true_tau.to_numpy()


# --------------------------------------------------------------------------
# Sample weights — the Bayesian bootstrap is silently useless without them
# --------------------------------------------------------------------------

@pytest.mark.parametrize("learner_cls", [SLearner, TLearner])
def test_learners_actually_use_sample_weight(synthetic, learner_cls):
    """A learner that ignores weights yields a zero-width posterior.

    That failure is invisible downstream: every draw agrees, the credible
    intervals collapse, and the model looks maximally confident.
    """
    X, t, y, _ = synthetic
    rng = np.random.default_rng(1)
    w1 = rng.dirichlet(np.ones(len(y))) * len(y)
    w2 = rng.dirichlet(np.ones(len(y))) * len(y)

    a = learner_cls().fit(X, t, y, sample_weight=w1).predict_uplift(X)
    b = learner_cls().fit(X, t, y, sample_weight=w2).predict_uplift(X)
    assert not np.allclose(a, b), f"{learner_cls.__name__} ignores sample_weight"


def test_posterior_has_non_degenerate_spread(synthetic):
    X, t, y, _ = synthetic
    model = BayesianBootstrapUplift(n_draws=25).fit(X, t, y)
    post = model.predict_posterior(X)
    assert post.shape == (25, len(y))
    assert model.summarize(X, post).std.mean() > 1e-6


def test_posterior_mean_tracks_the_planted_effect(synthetic):
    X, t, y, true_tau = synthetic
    model = BayesianBootstrapUplift(n_draws=25).fit(X, t, y)
    summary = model.summarize(X)
    assert np.corrcoef(summary.mean, true_tau)[0, 1] > 0.5


def test_credible_interval_brackets_its_own_mean(synthetic):
    X, t, y, _ = synthetic
    s = BayesianBootstrapUplift(n_draws=25).fit(X, t, y).summarize(X)
    assert (s.lower <= s.mean).all() and (s.mean <= s.upper).all()


def test_inflation_widens_without_moving_the_ranking():
    """The correction must change confidence only — never the targeting order."""
    rng = np.random.default_rng(2)
    post = rng.normal(0.05, 0.01, size=(100, 500))
    wide = inflate_posterior(post, 2.0)
    np.testing.assert_allclose(post.mean(axis=0), wide.mean(axis=0), atol=1e-12)
    assert wide.std(axis=0).mean() == pytest.approx(2 * post.std(axis=0).mean(), rel=1e-6)
    np.testing.assert_array_equal(
        np.argsort(-post.mean(axis=0)), np.argsort(-wide.mean(axis=0))
    )


def test_decision_rule_tightens_as_contact_gets_expensive():
    rng = np.random.default_rng(3)
    post = rng.normal(0.05, 0.02, size=(200, 1000))
    cheap = decision_rule(post, 100.0, 0.10, confidence=0.8)["contact"].mean()
    dear = decision_rule(post, 100.0, 8.00, confidence=0.8)["contact"].mean()
    assert cheap > dear
    assert decision_rule(post, 100.0, 0.10)["break_even_uplift"] == pytest.approx(0.001)


def test_calibration_detects_a_deliberately_overconfident_posterior(synthetic):
    """Shrinking a posterior to near-zero width must be reported as over-confident."""
    X, t, y, _ = synthetic
    model = BayesianBootstrapUplift(n_draws=40).fit(X, t, y)
    post = model.predict_posterior(X)
    over = inflate_posterior(post, 0.01)
    summary = coverage_summary(group_coverage(y, t, over), 0.9)
    assert summary["z_sd"] > 1.0
    assert "over-confident" in summary["verdict"]


# --------------------------------------------------------------------------
# Experiment design
# --------------------------------------------------------------------------

def test_sample_size_matches_a_hand_checked_case():
    """Guards the formula against a silent constant-factor error.

    p1=0.10 vs p2=0.12 at alpha=0.05, power=0.80 is a textbook case needing
    roughly 3,800-3,900 per arm.
    """
    d = sample_size_two_proportions(0.10, 0.12, alpha=0.05, power=0.8)
    assert 3_500 <= d["n_control"] <= 4_300


def test_smaller_effects_need_quadratically_more_data():
    big = sample_size_two_proportions(0.10, 0.12)["n_total"]
    small = sample_size_two_proportions(0.10, 0.11)["n_total"]
    # Halving the effect should roughly quadruple the requirement.
    assert 3.4 <= small / big <= 4.6


def test_more_power_and_tighter_alpha_both_cost_data():
    base = sample_size_two_proportions(0.10, 0.12, alpha=0.05, power=0.8)["n_total"]
    assert sample_size_two_proportions(0.10, 0.12, alpha=0.05, power=0.95)["n_total"] > base
    assert sample_size_two_proportions(0.10, 0.12, alpha=0.01, power=0.8)["n_total"] > base


def test_mde_inverts_the_sample_size_calculation():
    """The two functions must agree, or one of them is wrong."""
    n = sample_size_two_proportions(0.10, 0.12, alpha=0.05, power=0.8)["n_total"]
    mde = mde_for_sample_size(n, 0.10, alpha=0.05, power=0.8)["mde_absolute"]
    assert mde == pytest.approx(0.02, rel=0.12)


def test_unbalanced_split_costs_power():
    balanced = sample_size_two_proportions(0.10, 0.12, ratio=1.0)["n_total"]
    lopsided = sample_size_two_proportions(0.10, 0.12, ratio=9.0)["n_total"]
    assert lopsided > balanced


def test_withholding_lowers_response_when_no_sleeping_dogs(synthetic_positive):
    """With every effect positive, dropping customers can only lose conversions.

    This is the trap the profit-based design exists to avoid: a response-rate
    test on such a population is guaranteed to make targeting look harmful,
    because the gains are entirely in the costs avoided.
    """
    X, t, y, true_tau = synthetic_positive
    d = design_policy_test(y, t, true_tau, depth=0.5)
    assert d["expected_difference"] < 0


def test_withholding_raises_response_when_sleeping_dogs_exist(synthetic):
    """With real negative effects, dropping the bottom half *helps* response.

    The mirror of the test above, and the reason the sign is a property of the
    population rather than an arithmetic certainty.
    """
    X, t, y, true_tau = synthetic
    d = design_policy_test(y, t, true_tau, depth=0.5)
    assert d["expected_difference"] > 0
    assert d["effect_among_dropped"] < 0


def test_shallower_targeting_makes_the_response_test_cheaper(synthetic_positive):
    X, t, y, true_tau = synthetic_positive
    shallow = design_policy_test(y, t, true_tau, depth=0.2)
    deep = design_policy_test(y, t, true_tau, depth=0.9)
    assert abs(shallow["expected_difference"]) > abs(deep["expected_difference"])
    assert shallow["n_total"] < deep["n_total"]


def test_profit_test_prefers_targeting_when_contact_is_expensive(synthetic_positive):
    """With no sleeping dogs, the sign of the profit difference follows the cost.

    Near-free contact makes treating everyone optimal; expensive contact makes
    withholding pay. (With sleeping dogs present, targeting wins at any cost,
    which is why this uses the all-positive fixture.)
    """
    X, t, y, true_tau = synthetic_positive
    cheap = design_profit_test(y, t, true_tau, 0.5, value_per_conversion=100.0, cost_per_contact=0.01)
    dear = design_profit_test(y, t, true_tau, 0.5, value_per_conversion=100.0, cost_per_contact=20.0)
    assert not cheap["b_is_better"], "with near-free contact, treating everyone should win"
    assert dear["b_is_better"], "with expensive contact, targeting should win"
    assert dear["n_total"] < cheap["n_total"]


def test_profit_test_honours_an_observed_revenue_column(synthetic):
    """Passing observed revenue must change the answer, not be silently ignored."""
    X, t, y, true_tau = synthetic
    rng = np.random.default_rng(4)
    revenue = y * rng.lognormal(3.0, 1.0, len(y))
    modeled = design_profit_test(y, t, true_tau, 0.5, 100.0, 0.5)
    observed = design_profit_test(y, t, true_tau, 0.5, 100.0, 0.5, value=revenue)
    assert modeled["basis"] == "modeled" and observed["basis"] == "observed"
    assert modeled["difference"] != observed["difference"]


def test_detecting_a_smaller_decay_needs_a_bigger_holdout():
    big_drop = design_decay_monitor(0.06, 0.10, detectable_decay=0.5)
    small_drop = design_decay_monitor(0.06, 0.10, detectable_decay=0.2)
    assert small_drop["n_total_holdout"] > big_drop["n_total_holdout"]


def test_power_curve_is_monotone_and_bounded():
    curve = power_curve(0.10, 0.02, np.array([100, 1_000, 10_000, 100_000]))
    assert curve["power"].is_monotonic_increasing
    assert (curve["power"] >= 0).all() and (curve["power"] <= 1).all()
    assert curve["power"].iloc[-1] > 0.95


def test_duration_scales_inversely_with_traffic():
    d = duration_table(100_000, [1_000, 10_000])
    assert d.loc[0, "days"] == pytest.approx(10 * d.loc[1, "days"])
