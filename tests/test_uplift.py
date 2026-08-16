"""Tests for the parts where a silent error would be invisible.

Uplift code fails quietly: get the Qini rescaling wrong and you still get a
plausible-looking curve and a plausible-looking number. So the tests here are
built around cases with a known answer — synthetic data with a treatment effect
that is planted rather than estimated, and metric identities that must hold
exactly regardless of the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uplift.data import average_treatment_effect, randomization_check
from uplift.evaluate import auuc, decile_table, qini_coefficient, qini_curve
from uplift.learners import SLearner, TLearner, XLearner
from uplift.monitoring import UpliftDriftMonitor, population_stability_index, realized_effect
from uplift.segments import classify, derive_cuts, summarize
from uplift.simulate import Economics, compare_policies, simulate_policy


@pytest.fixture
def synthetic():
    """A randomized experiment where the true effect is known by construction.

    Effect is driven entirely by ``x0``: strongly positive for high x0, negative
    for low x0. Any correct implementation must recover that ordering, and the
    negative region gives the sleeping-dog logic something real to find.
    """
    rng = np.random.default_rng(0)
    n = 20_000
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"x{i}" for i in range(4)])
    t = rng.binomial(1, 0.5, n)
    base = 0.2 + 0.05 * X["x1"]
    true_tau = 0.25 * X["x0"]
    p = np.clip(base + t * true_tau, 0.01, 0.99)
    y = rng.binomial(1, p)
    return X, t, y, true_tau.to_numpy()


# --------------------------------------------------------------------------
# Metric identities: these must hold for any input, so they catch sign flips
# and normalization mistakes that eyeballing a curve would not.
# --------------------------------------------------------------------------

def test_qini_of_random_ranking_is_near_zero(synthetic):
    X, t, y, _ = synthetic
    rng = np.random.default_rng(1)
    coefs = [qini_coefficient(qini_curve(y, t, rng.normal(size=len(y)))) for _ in range(20)]
    assert abs(np.mean(coefs)) < 0.05, "a random ranking should score ~0 Qini"


def test_qini_of_oracle_beats_random(synthetic):
    X, t, y, true_tau = synthetic
    rng = np.random.default_rng(2)
    oracle = qini_coefficient(qini_curve(y, t, true_tau))
    random = qini_coefficient(qini_curve(y, t, rng.normal(size=len(y))))
    assert oracle > random + 0.1
    assert oracle > 0.2


def test_reversing_the_ranking_flips_the_sign(synthetic):
    """The clearest test of the metric's orientation."""
    X, t, y, true_tau = synthetic
    good = qini_coefficient(qini_curve(y, t, true_tau))
    bad = qini_coefficient(qini_curve(y, t, -true_tau))
    assert good > 0 > bad
    assert bad == pytest.approx(-good, rel=0.35)


def test_qini_curve_endpoint_equals_overall_effect(synthetic):
    """At 100% depth the Qini curve must equal the rescaled total effect."""
    X, t, y, true_tau = synthetic
    curve = qini_curve(y, t, true_tau)
    n_t, n_c = t.sum(), (1 - t).sum()
    expected = y[t == 1].sum() - y[t == 0].sum() * n_t / n_c
    assert curve["qini"].iloc[-1] == pytest.approx(expected, rel=1e-6)


def test_qini_is_undefined_when_there_is_no_effect_to_allocate():
    """No campaign effect must report NaN, not a flattering number.

    The coefficient divides by the total incremental outcome. With no true
    effect that denominator is noise, and the ratio happily returns values
    above 1.0 — which would read as a near-perfect model on an experiment where
    nothing happened.
    """
    rng = np.random.default_rng(3)
    n = 20_000
    t = rng.binomial(1, 0.9, n)  # 9:1 split, to also exercise the rescaling
    y = rng.binomial(1, 0.2, n)  # outcome independent of treatment
    assert np.isnan(qini_coefficient(qini_curve(y, t, rng.normal(size=n))))


def test_qini_handles_unbalanced_arms():
    """A 9:1 split must not inflate the coefficient; that is what rescaling is for."""
    rng = np.random.default_rng(13)
    n = 60_000
    t = rng.binomial(1, 0.9, n)
    tau_true = 0.25 * rng.normal(size=n)
    y = rng.binomial(1, np.clip(0.2 + t * (tau_true + 0.05), 0.01, 0.99))
    oracle = qini_coefficient(qini_curve(y, t, tau_true))
    random_rank = qini_coefficient(qini_curve(y, t, rng.normal(size=n)))
    assert oracle > random_rank + 0.1
    assert oracle > 0
    # Deliberately no upper bound of 1: with negative-effect mass present, a
    # ranking that avoids it beats treating everyone, which is the denominator.
    assert abs(random_rank) < 0.15


def test_auuc_of_constant_scores_is_about_one(synthetic):
    """Constant predictions rank nothing, so the uplift curve is the diagonal."""
    X, t, y, _ = synthetic
    curve = qini_curve(y, t, np.zeros(len(y)))
    assert auuc(curve) == pytest.approx(1.0, abs=0.15)


# --------------------------------------------------------------------------
# Learners
# --------------------------------------------------------------------------

@pytest.mark.parametrize("learner_cls", [SLearner, TLearner, XLearner])
def test_learners_recover_planted_effect_ordering(synthetic, learner_cls):
    """Each learner must correlate with the true effect it was built to find."""
    X, t, y, true_tau = synthetic
    split = 15_000
    learner = learner_cls().fit(X.iloc[:split], t[:split], y[:split])
    pred = learner.predict_uplift(X.iloc[split:])
    corr = np.corrcoef(pred, true_tau[split:])[0, 1]
    assert corr > 0.5, f"{learner_cls.__name__} correlation with truth was {corr:.3f}"


def test_learners_accept_numpy_and_dataframe_alike(synthetic):
    X, t, y, _ = synthetic
    learner = TLearner().fit(X, t, y)
    from_frame = learner.predict_uplift(X.head(50))
    from_array = learner.predict_uplift(X.head(50).to_numpy())
    np.testing.assert_allclose(from_frame, from_array)


def test_s_learner_finds_no_effect_when_there_is_none():
    """Guards against a learner that manufactures uplift out of noise."""
    rng = np.random.default_rng(4)
    n = 10_000
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    t = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, 0.3, n)  # outcome ignores treatment entirely
    tau = SLearner().fit(X, t, y).predict_uplift(X)
    assert abs(tau.mean()) < 0.02


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------

def test_segments_are_stable_under_batching(synthetic):
    """A customer's segment must not depend on who else is in the request.

    This is the reason the cuts are frozen at training time rather than derived
    per batch; without it, the same customer scored alone and scored in a batch
    could get different recommendations.
    """
    X, t, y, true_tau = synthetic
    p_control = np.clip(0.2 + 0.05 * X["x1"].to_numpy(), 0, 1)
    cuts = derive_cuts(true_tau, p_control)
    full = classify(true_tau, p_control, **cuts)
    chunk = classify(true_tau[:100], p_control[:100], **cuts)
    np.testing.assert_array_equal(full[:100], chunk)


def test_sleeping_dogs_are_only_negative_predictions(synthetic):
    X, t, y, true_tau = synthetic
    p_control = np.clip(0.2 + 0.05 * X["x1"].to_numpy(), 0, 1)
    segs = classify(true_tau, p_control, **derive_cuts(true_tau, p_control))
    assert (true_tau[segs == "sleeping_dog"] < 0).all()
    assert (segs == "sleeping_dog").sum() > 0, "planted negative effects should be found"


def test_segment_summary_recovers_planted_ordering(synthetic):
    """Persuadables must show a larger observed effect than sleeping dogs."""
    X, t, y, true_tau = synthetic
    p_control = np.clip(0.2 + 0.05 * X["x1"].to_numpy(), 0, 1)
    segs = classify(true_tau, p_control, **derive_cuts(true_tau, p_control))
    summary = summarize(segs, y, t).set_index("segment")
    assert summary.loc["persuadable", "observed_uplift"] > summary.loc["sleeping_dog", "observed_uplift"]
    assert summary.loc["sleeping_dog", "observed_uplift"] < 0


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def test_simulation_never_invents_effect(synthetic):
    """At full depth every policy must report the same total incremental effect."""
    X, t, y, true_tau = synthetic
    econ = Economics(100.0, 0.1)
    rng = np.random.default_rng(5)
    a = simulate_policy(true_tau, y, t, econ, depths=[1.0])
    b = simulate_policy(rng.normal(size=len(y)), y, t, econ, depths=[1.0])
    assert a["incremental_conversions"].iloc[0] == pytest.approx(
        b["incremental_conversions"].iloc[0], rel=1e-9
    )


def test_better_ranking_wins_at_shallow_depth(synthetic):
    X, t, y, true_tau = synthetic
    econ = Economics(100.0, 0.1)
    rng = np.random.default_rng(6)
    comp = compare_policies({"uplift": true_tau, "noise": rng.normal(size=len(y))}, y, t, econ)
    at_20 = comp[np.isclose(comp["depth"], 0.2)].set_index("policy")
    assert at_20.loc["uplift", "incremental_conversions"] > at_20.loc["noise", "incremental_conversions"]
    assert at_20.loc["uplift", "incremental_conversions"] > at_20.loc["random", "incremental_conversions"]


def test_cost_shrinks_the_optimal_campaign(synthetic):
    """Raising the cost of a contact must never widen the optimal campaign."""
    X, t, y, true_tau = synthetic
    depths = np.arange(0.1, 1.01, 0.1)
    cheap = simulate_policy(true_tau, y, t, Economics(100.0, 0.01), depths=depths)
    dear = simulate_policy(true_tau, y, t, Economics(100.0, 5.0), depths=depths)
    assert dear.loc[dear["profit_modeled"].idxmax(), "depth"] <= cheap.loc[
        cheap["profit_modeled"].idxmax(), "depth"
    ]


# --------------------------------------------------------------------------
# Data checks and monitoring
# --------------------------------------------------------------------------

def test_randomization_check_flags_a_broken_experiment():
    """A confounded assignment must fail the balance check that clean data passes."""
    rng = np.random.default_rng(7)
    n = 10_000
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    good = rng.binomial(1, 0.5, n)
    assert randomization_check(X, good)["balanced"].all()

    # Assignment now depends on a covariate — exactly what unconfoundedness rules out.
    bad = (X["a"] + rng.normal(0, 0.3, n) > 0).astype(int).to_numpy()
    assert not randomization_check(X, bad)["balanced"].all()


def test_ate_confidence_interval_covers_planted_effect():
    rng = np.random.default_rng(8)
    n = 50_000
    t = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, 0.2 + 0.05 * t)
    ate = average_treatment_effect(y, t)
    assert ate["ate_ci_low"] < 0.05 < ate["ate_ci_high"]


def test_qini_ci_degrades_instead_of_raising():
    """Zero bootstrap draws must yield NaN bounds, not an IndexError.

    The dashboard asks for a curve without a CI; an empty percentile call there
    took down the whole page.
    """
    from uplift.evaluate import evaluate, qini_with_ci

    rng = np.random.default_rng(14)
    n = 5_000
    t = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, 0.2 + 0.05 * t)
    out = qini_with_ci(y, t, rng.normal(size=n), n_boot=0)
    assert np.isnan(out["qini_ci_low"]) and out["n_boot_valid"] == 0
    assert evaluate(y, t, rng.normal(size=n), n_boot=0)["curve"] is not None


def test_changing_the_outcome_changes_the_split(synthetic):
    """`prepare` stratifies on the outcome, so each outcome gets its own split.

    Pinned because the consequence is easy to miss and fails silently: pairing
    one outcome's y against another split's t lines up mismatched rows and
    shrinks the estimated effect toward zero, while still returning a
    perfectly plausible-looking number.
    """
    import pandas as pd

    from uplift.config import HILLSTROM
    from uplift.data import prepare

    a = prepare(HILLSTROM, outcome="visit")
    b = prepare(HILLSTROM, outcome="conversion")
    t_a = np.concatenate([a.t_train, a.t_test])
    t_b = np.concatenate([b.t_train, b.t_test])
    assert not np.array_equal(t_a, t_b), (
        "splits coincided; this test can no longer detect the mismatch it guards"
    )

    y_b = np.concatenate([b.y_train, b.y_test])
    correct = average_treatment_effect(y_b, t_b)["ate"]
    mismatched = average_treatment_effect(y_b, t_a)["ate"]
    assert correct > 0.004
    assert mismatched < correct / 3, "mismatched pairing should dilute the effect"


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(9)
    x = rng.normal(size=5000)
    assert population_stability_index(x, x) == pytest.approx(0, abs=1e-3)


def test_monitor_catches_effect_decay_but_not_ordinary_noise():
    """The headline claim of the monitoring layer, tested directly."""
    rng = np.random.default_rng(10)
    reference = rng.normal(0.05, 0.02, 40_000)
    monitor = UpliftDriftMonitor(reference)

    for _ in range(5):
        normal_batch = rng.normal(0.05, 0.02, 2000)
        assert monitor.check(normal_batch).status == "OK"

    decayed = rng.normal(0.05, 0.02, 2000) * 0.5
    assert monitor.check(decayed, batch="decay").status == "ALERT"


def test_monitor_band_scales_with_batch_size():
    """A small batch must be given more slack than a large one."""
    reference = np.random.default_rng(11).normal(0.05, 0.02, 40_000)
    monitor = UpliftDriftMonitor(reference)
    small_lo, small_hi = monitor.mean_band(100)
    large_lo, large_hi = monitor.mean_band(100_000)
    assert (small_hi - small_lo) > (large_hi - large_lo)


def test_realized_effect_matches_planted_truth():
    rng = np.random.default_rng(12)
    n = 40_000
    tau_true = rng.uniform(0, 0.2, n)
    t = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, np.clip(0.2 + t * tau_true, 0, 1))
    out = realized_effect(y, t, tau_true, top_frac=0.3)
    assert out["observed_uplift"] == pytest.approx(out["predicted_uplift"], abs=0.03)


def test_decile_table_is_monotone_for_a_perfect_ranking(synthetic):
    X, t, y, true_tau = synthetic
    dec = decile_table(y, t, true_tau)
    top, bottom = dec.iloc[0]["observed_uplift"], dec.iloc[-1]["observed_uplift"]
    assert top > bottom
    assert np.corrcoef(dec["decile"], dec["observed_uplift"])[0, 1] < -0.8
