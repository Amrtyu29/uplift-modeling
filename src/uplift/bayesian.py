"""Uncertainty quantification on individual treatment effects.

A point estimate of tau(x) is a weak basis for spending money. Two customers
with an identical predicted uplift of +0.05 are not equivalent if one estimate
rests on 8,000 similar training rows and the other on 40 — but a single number
cannot tell them apart, and the targeting policy will treat them identically.

This module puts a posterior on tau(x) instead, which makes three things
possible that a point estimate cannot support:

1. **Confidence-aware targeting.** Contact only where P(tau x value > cost)
   clears a stated bar, rather than wherever the point estimate happens to.
2. **An honest reliability check.** Credible intervals make a falsifiable
   claim — a 90% interval should contain the truth 90% of the time — and
   ``group_coverage`` tests exactly that on randomized holdout data.
3. **Knowing where the model is guessing.** High-variance regions are where
   more experimentation is worth buying.

Method: the **Bayesian bootstrap** (Rubin, 1981). Each posterior draw reweights
the training rows by weights drawn from a Dirichlet(1, ..., 1) and refits the
base learner. This is a genuine posterior over the empirical distribution of the
data, not an ad-hoc ensemble, and unlike a fully specified Bayesian model it
imposes no likelihood or prior on the outcome — which matters here, because the
quantity of interest is a difference of two conditional probabilities with no
natural conjugate form.

The trade-off, stated plainly: this captures uncertainty from *sampling* the
data. It does not capture uncertainty about the model class being right, and it
cannot capture anything about unobserved confounding. On randomized data the
second is not an issue; the first is real and means these intervals are a lower
bound on total uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RANDOM_STATE
from .learners import BaseLearner, SLearner


@dataclass
class PosteriorSummary:
    """Per-customer posterior over the treatment effect."""

    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    prob_positive: np.ndarray
    credible_level: float

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "uplift_mean": self.mean,
                "uplift_std": self.std,
                "uplift_lower": self.lower,
                "uplift_upper": self.upper,
                "prob_positive": self.prob_positive,
            }
        )


class BayesianBootstrapUplift:
    """Posterior over tau(x) via Bayesian-bootstrap resampling of a base learner.

    Works with any learner in this package, since it only needs ``fit`` with
    sample weights and ``predict_uplift``. The default base is the S-learner —
    on Hillstrom it is both the best-ranking and the cheapest to refit, and this
    class refits it ``n_draws`` times.
    """

    def __init__(
        self,
        base_learner_factory=SLearner,
        n_draws: int = 200,
        credible_level: float = 0.9,
        random_state: int = RANDOM_STATE,
    ):
        self.base_learner_factory = base_learner_factory
        self.n_draws = n_draws
        self.credible_level = credible_level
        self.random_state = random_state
        self.learners_: list[BaseLearner] = []

    def fit(self, X, t: np.ndarray, y: np.ndarray) -> "BayesianBootstrapUplift":
        """Refit the base learner once per posterior draw.

        Cost is linear in ``n_draws`` — 200 draws is 200 fits — so on large data
        lower it rather than waiting. The posterior mean stabilizes quickly;
        it is the tail percentiles that need the draws.
        """
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        self.learners_ = []
        for _ in range(self.n_draws):
            # Bayesian bootstrap weights: a smooth reweighting rather than the
            # classical bootstrap's integer resampling, which keeps every row in
            # play and avoids draws that lose a treatment arm entirely.
            #
            # Normalized iid Exponential(1) draws are exactly Dirichlet(1,...,1)
            # and far cheaper than np.random.dirichlet at this length, which
            # builds the whole simplex sample at once and loses precision when n
            # runs to the millions.
            w = rng.exponential(size=n)
            w *= n / w.sum()
            learner = self.base_learner_factory()
            learner.fit(X, t, y, sample_weight=w)
            self.learners_.append(learner)
        return self

    def predict_posterior(self, X) -> np.ndarray:
        """Posterior draws of tau, shape ``(n_draws, n_customers)``."""
        if not self.learners_:
            raise RuntimeError("call fit() first")
        return np.vstack([learner.predict_uplift(X) for learner in self.learners_])

    def summarize(self, X, posterior: np.ndarray | None = None) -> PosteriorSummary:
        post = self.predict_posterior(X) if posterior is None else posterior
        alpha = (1 - self.credible_level) / 2
        return PosteriorSummary(
            mean=post.mean(axis=0),
            std=post.std(axis=0, ddof=1),
            lower=np.percentile(post, 100 * alpha, axis=0),
            upper=np.percentile(post, 100 * (1 - alpha), axis=0),
            prob_positive=(post > 0).mean(axis=0),
            credible_level=self.credible_level,
        )

    def prob_above(self, threshold: float, X=None, posterior: np.ndarray | None = None) -> np.ndarray:
        """P(tau > threshold) per customer — the quantity a decision needs.

        With a contact cost c and conversion value v, the break-even effect is
        c / v. Targeting on P(tau > c/v) asks "how sure are we this customer is
        worth contacting", which is the question, rather than "is the single
        best guess above the line".
        """
        post = self.predict_posterior(X) if posterior is None else posterior
        return (post > threshold).mean(axis=0)


def decision_rule(
    posterior: np.ndarray,
    value_per_conversion: float,
    cost_per_contact: float,
    confidence: float = 0.8,
) -> dict:
    """Confidence-aware targeting: contact only when the bar is cleared.

    Returns the mask plus the break-even threshold it used, so the caller can
    report the rule rather than just its output.
    """
    threshold = cost_per_contact / value_per_conversion
    prob = (posterior > threshold).mean(axis=0)
    return {
        "break_even_uplift": threshold,
        "prob_above_break_even": prob,
        "contact": prob >= confidence,
        "confidence": confidence,
    }


def group_coverage(
    y: np.ndarray,
    t: np.ndarray,
    posterior: np.ndarray,
    bins: int = 10,
    credible_level: float = 0.9,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Do the credible intervals actually contain the truth?

    Individual effects are unobservable, so per-customer coverage cannot be
    tested. Group means can be: for each bin of customers, the posterior implies
    a distribution over that group's *average* effect (average the draws within
    the group), and randomized data supplies an unbiased observed estimate of
    the same quantity. A calibrated model puts the observed value inside its
    interval about ``credible_level`` of the time.

    One subtlety decides whether this test is meaningful. The "truth" being
    compared against is itself an estimate: with ~2,000 customers per bin split
    across two arms, the observed uplift carries a standard error of roughly
    0.01-0.03, which is the same size as the posterior interval. Asking whether
    a noisy point lands inside a narrow interval therefore fails almost always,
    even for a perfectly calibrated model.

    So the primary statistic reported is the standardized residual

        z = (observed - posterior_mean) / sqrt(posterior_var + observed_se^2)

    which puts both sources of uncertainty on the same footing. If the posterior
    is calibrated, these have standard deviation 1. Above 1 means over-confident,
    below 1 means conservative. Raw coverage is kept alongside it, but it is the
    weaker statistic and is reported as such.
    """
    y, t = np.asarray(y, float), np.asarray(t, int)
    mean_tau = posterior.mean(axis=0)

    rng = np.random.default_rng(random_state)
    order = np.lexsort((rng.random(len(mean_tau)), -mean_tau))
    ranks = np.empty(len(mean_tau), int)
    ranks[order] = np.arange(len(mean_tau))
    group = np.minimum((ranks * bins) // len(mean_tau), bins - 1)

    alpha = (1 - credible_level) / 2
    rows = []
    for g in range(bins):
        m = group == g
        yt, yc = y[m & (t == 1)], y[m & (t == 0)]
        if len(yt) == 0 or len(yc) == 0:
            continue
        p1, p0 = yt.mean(), yc.mean()
        observed = p1 - p0
        se = np.sqrt(p1 * (1 - p1) / len(yt) + p0 * (1 - p0) / len(yc))

        # Posterior over the group's mean effect: average within the group per
        # draw, which preserves the correlation between customers in a draw.
        group_draws = posterior[:, m].mean(axis=1)
        lo, hi = np.percentile(group_draws, [100 * alpha, 100 * (1 - alpha)])
        post_mean = float(group_draws.mean())
        post_sd = float(group_draws.std(ddof=1))
        total_sd = np.sqrt(post_sd**2 + se**2)
        rows.append(
            {
                "bin": g + 1,
                "n": int(m.sum()),
                "posterior_mean": post_mean,
                "posterior_sd": post_sd,
                "credible_low": float(lo),
                "credible_high": float(hi),
                "observed_uplift": float(observed),
                "observed_se": float(se),
                # Standardized residual: the statistic that actually tests
                # calibration, since it prices in the observation's own noise.
                "z": float((observed - post_mean) / total_sd) if total_sd > 0 else np.nan,
                "covered": bool(lo <= observed <= hi),
                "covered_within_noise": bool(lo - 1.96 * se <= observed <= hi + 1.96 * se),
            }
        )
    return pd.DataFrame(rows)


def inflate_posterior(posterior: np.ndarray, factor: float) -> np.ndarray:
    """Widen every posterior around its own mean by ``factor``.

    The Bayesian bootstrap only propagates uncertainty from *resampling the
    data*. It says nothing about whether the model class is right, so when the
    standardized residuals come out with sd > 1, the shortfall is real and the
    intervals understate what is actually not known.

    Scaling the spread by the measured sd(z) is the minimal honest correction:
    it leaves the point estimate — and therefore the entire targeting ranking —
    untouched, and only widens the stated confidence. The factor must be
    estimated on data not used to evaluate it, or it is guaranteed to look
    perfect by construction.
    """
    mean = posterior.mean(axis=0, keepdims=True)
    return mean + (posterior - mean) * factor


def coverage_summary(coverage: pd.DataFrame, credible_level: float = 0.9) -> dict:
    """Headline calibration numbers, judged on the standardized residuals.

    The verdict comes from sd(z), not from raw coverage. With ten bins the
    sampling error on sd(z) is itself about 1/sqrt(2*10) ~ 0.22, so the bands
    below are deliberately wide — calling a model miscalibrated off ten noisy
    bins would be exactly the overconfidence this module exists to avoid.
    """
    z = coverage["z"].dropna().to_numpy()
    z_sd = float(np.std(z, ddof=1)) if len(z) > 1 else float("nan")
    z_mean = float(np.mean(z)) if len(z) else float("nan")
    rate = float(coverage["covered"].mean())
    width = float((coverage["credible_high"] - coverage["credible_low"]).mean())

    if not np.isfinite(z_sd):
        verdict = "not enough bins to judge calibration"
    elif z_sd > 1.5:
        verdict = "over-confident — residuals are larger than the posterior claims"
    elif z_sd < 0.6:
        verdict = "conservative — posterior is wider than the errors require"
    else:
        verdict = "consistent with calibration"

    bias = ""
    if np.isfinite(z_mean) and abs(z_mean) > 1.0:
        bias = (
            " Systematic bias: observed effects run "
            + ("above" if z_mean > 0 else "below")
            + " the posterior mean across bins, which is a ranking/shrinkage issue rather"
            " than an interval-width one."
        )

    return {
        "nominal_level": credible_level,
        "z_sd": z_sd,
        "z_mean": z_mean,
        "verdict": verdict + bias,
        # Reported for completeness, but the weaker statistic: it compares a
        # noisy observation against a narrow interval and will read low even
        # when the posterior is fine.
        "raw_coverage": rate,
        "coverage_within_observation_noise": float(coverage["covered_within_noise"].mean()),
        "mean_interval_width": width,
        "n_bins": int(len(coverage)),
    }
