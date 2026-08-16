"""Designing the experiment that would validate this model in production.

Everything else in this project estimates what a policy *would have* earned on
historical randomized data. That is the right way to choose a policy. It is not
the same as having run one, and the gap between those two statements is where
most "our model delivered X% lift" claims quietly fall apart.

This module sizes the experiment that closes the gap. Three distinct questions,
which need different designs and are routinely confused:

1. **Does the treatment work at all?** Treated vs. control. Usually already
   answered, and the easiest to power.
2. **Does targeting by uplift beat the current blanket campaign?** Two *policies*
   compared head to head. This is the one that justifies the project, and it is
   far more expensive to power than people expect — because both arms contain
   mostly the same customers receiving the same treatment, so the difference
   being detected is small even when the model is genuinely good.
3. **Has the effect decayed?** An always-on randomized holdout, sized to notice
   a drop before it has cost real money.

The recurring lesson from the numbers below: powering (2) is the hard one, and
finding that out during design is much cheaper than finding out after a
six-week inconclusive test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def sample_size_two_proportions(
    p_control: float,
    p_treated: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    two_sided: bool = True,
) -> dict:
    """Per-arm sample size to detect a difference between two rates.

    Standard normal-approximation formula. ``ratio`` is n_treated / n_control,
    which matters because an unbalanced split costs power: the effective sample
    size is governed by the harmonic mean of the two arms, so a 90/10 split
    needs far more total traffic than 50/50 to reach the same power.
    """
    if not 0 < p_control < 1 or not 0 < p_treated < 1:
        raise ValueError("rates must be strictly between 0 and 1")
    delta = abs(p_treated - p_control)
    if delta == 0:
        return {"n_control": float("inf"), "n_treated": float("inf"), "n_total": float("inf"),
                "mde_absolute": 0.0}

    z_alpha = stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)
    z_beta = stats.norm.ppf(power)
    var = p_control * (1 - p_control) + p_treated * (1 - p_treated) / ratio
    n_control = ((z_alpha + z_beta) ** 2 * var) / delta**2

    n_control = int(np.ceil(n_control))
    n_treated = int(np.ceil(n_control * ratio))
    return {
        "n_control": n_control,
        "n_treated": n_treated,
        "n_total": n_control + n_treated,
        "mde_absolute": delta,
        "mde_relative": delta / p_control,
        "alpha": alpha,
        "power": power,
        "ratio": ratio,
    }


def mde_for_sample_size(
    n_total: int,
    p_control: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
) -> dict:
    """The smallest effect a given amount of traffic can reliably detect.

    The inverse of the question above, and usually the more useful one: traffic
    is fixed by the business, so the real decision is whether the effect you
    hope for is even inside the detectable range. Solved numerically because
    the variance term depends on the unknown treated rate.
    """
    n_control = n_total / (1 + ratio)
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    def required(delta: float) -> float:
        p_t = min(max(p_control + delta, 1e-9), 1 - 1e-9)
        var = p_control * (1 - p_control) + p_t * (1 - p_t) / ratio
        return ((z_alpha + z_beta) ** 2 * var) / delta**2

    lo, hi = 1e-6, max(0.5, p_control)
    for _ in range(200):
        mid = (lo + hi) / 2
        if required(mid) > n_control:
            lo = mid
        else:
            hi = mid
    delta = (lo + hi) / 2
    return {
        "n_total": int(n_total),
        "mde_absolute": float(delta),
        "mde_relative": float(delta / p_control),
        "p_control": p_control,
        "alpha": alpha,
        "power": power,
    }


def design_policy_test(
    y: np.ndarray,
    t: np.ndarray,
    uplift: np.ndarray,
    depth: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict:
    """Size the test that compares uplift targeting against a blanket campaign.

    The two arms are policies, not treatments:

      A (incumbent): contact everyone
      B (proposed) : contact the top ``depth`` by predicted uplift

    The subtlety that makes this expensive: the arms overlap heavily. Every
    customer in the targeted top slice is contacted under *both* policies, so
    they contribute nothing to the difference. The entire signal comes from the
    withheld remainder — the customers policy B drops — and the expected
    difference in overall response rate is therefore

        (1 - depth) x (effect among the dropped customers)

    which is small even when the model ranks well. The estimates below come
    from the randomized holdout, so they are measured rather than guessed.
    """
    y, t, uplift = np.asarray(y, float), np.asarray(t, int), np.asarray(uplift, float)
    n = len(y)
    k = max(1, int(round(depth * n)))
    order = np.argsort(-uplift)
    targeted = np.zeros(n, bool)
    targeted[order[:k]] = True

    # What policy A (treat everyone) produces, per customer.
    rate_a = y[t == 1].mean()

    # Policy B: targeted customers get the treated rate; dropped customers get
    # the control rate. Both read off the randomized holdout.
    yt_top = y[targeted & (t == 1)]
    yc_bottom = y[~targeted & (t == 0)]
    yt_bottom = y[~targeted & (t == 1)]
    if len(yt_top) == 0 or len(yc_bottom) == 0 or len(yt_bottom) == 0:
        return {"status": "insufficient_data"}

    rate_b = (k * yt_top.mean() + (n - k) * yc_bottom.mean()) / n
    dropped_effect = yt_bottom.mean() - yc_bottom.mean()

    design = sample_size_two_proportions(rate_a, rate_b, alpha=alpha, power=power)
    return {
        "depth": depth,
        "rate_policy_a_treat_all": float(rate_a),
        "rate_policy_b_targeted": float(rate_b),
        "expected_difference": float(rate_b - rate_a),
        "effect_among_dropped": float(dropped_effect),
        "share_dropped": float(1 - depth),
        "n_per_arm": design["n_control"],
        "n_total": design["n_total"],
        "detectable": bool(abs(rate_b - rate_a) > 0),
    }


def design_profit_test(
    y: np.ndarray,
    t: np.ndarray,
    uplift: np.ndarray,
    depth: float,
    value_per_conversion: float,
    cost_per_contact: float,
    alpha: float = 0.05,
    power: float = 0.8,
    value: np.ndarray | None = None,
) -> dict:
    """Size the policy test on **profit per customer**, which is the real outcome.

    Powering this test on response rate is a trap, and ``design_policy_test``
    exists partly to expose it. When every customer's effect is positive —
    which is the case on Hillstrom, where no subgroup shows a negative effect —
    withholding contact can only *lower* the overall response rate, so the
    response-rate difference comes out negative however good the model is. A
    test powered that way is set up to conclude that targeting hurts.

    The sign is a property of the population, not arithmetic: where genuine
    sleeping dogs exist, dropping them raises the response rate and the same
    test flatters the policy instead. Either way the metric is answering a
    different question from the one being asked.

    The quantity that actually decides the question is profit per customer:

        contacted     -> y x value - cost
        not contacted -> y x value

    Its variance is larger than a rate's, because it mixes outcome variance with
    the cost saving, so this is not a free improvement — but it is a test of the
    right hypothesis, which matters more than a cheaper test of the wrong one.

    ``value`` supplies observed per-customer revenue where the dataset has it.
    Passing it matters for more than precision: pricing a conversion at an
    assumed figure can flip which policy looks better, so a design built on one
    basis while the simulation used another will size a test for a conclusion
    the simulation never reached. Heavy-tailed real revenue also inflates the
    variance, and therefore the sample size, which is itself worth knowing.
    """
    y, t, uplift = np.asarray(y, float), np.asarray(t, int), np.asarray(uplift, float)
    n = len(y)
    k = max(1, int(round(depth * n)))
    order = np.argsort(-uplift)
    targeted = np.zeros(n, bool)
    targeted[order[:k]] = True

    # Revenue per customer: observed where available, otherwise conversions
    # priced at the assumed value.
    revenue = y * value_per_conversion if value is None else np.asarray(value, float)
    basis = "modeled" if value is None else "observed"

    rev_treated_all = revenue[t == 1]
    rev_top_treated = revenue[targeted & (t == 1)]
    rev_bottom_control = revenue[~targeted & (t == 0)]
    if len(rev_treated_all) == 0 or len(rev_top_treated) == 0 or len(rev_bottom_control) == 0:
        return {"status": "insufficient_data"}

    # Policy A: everyone contacted.
    profit_a = rev_treated_all - cost_per_contact
    mean_a, var_a = float(profit_a.mean()), float(profit_a.var(ddof=1))

    # Policy B: a mixture of contacted top and withheld bottom. Variance via
    # the law of total variance, so the between-component spread is included
    # rather than quietly dropped.
    p_top = rev_top_treated.mean() - cost_per_contact
    p_bot = rev_bottom_control.mean()
    v_top = float((rev_top_treated - cost_per_contact).var(ddof=1))
    v_bot = float(rev_bottom_control.var(ddof=1))
    w = depth
    mean_b = w * p_top + (1 - w) * p_bot
    var_b = (
        w * v_top + (1 - w) * v_bot
        + w * (p_top - mean_b) ** 2 + (1 - w) * (p_bot - mean_b) ** 2
    )

    diff = mean_b - mean_a
    if diff == 0:
        return {"status": "no_difference"}

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n_per_arm = int(np.ceil((z_alpha + z_beta) ** 2 * (var_a + var_b) / diff**2))

    return {
        "depth": depth,
        "basis": basis,
        "profit_per_customer_a": mean_a,
        "profit_per_customer_b": mean_b,
        "difference": float(diff),
        "sd_a": float(np.sqrt(var_a)),
        "sd_b": float(np.sqrt(var_b)),
        "n_per_arm": n_per_arm,
        "n_total": n_per_arm * 2,
        "b_is_better": bool(diff > 0),
    }


def design_decay_monitor(
    baseline_effect: float,
    control_rate: float,
    detectable_decay: float = 0.3,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict:
    """Size the always-on randomized holdout that catches effect decay.

    This is the piece teams skip, and it is the one that keeps every future
    retraining valid. Sizing it is the same two-proportion calculation, with the
    effect to detect being a *fraction* of the current effect rather than the
    effect itself — detecting a 30% decay needs materially more traffic than
    detecting the original effect did.
    """
    decayed_effect = baseline_effect * (1 - detectable_decay)
    # Comparing "effect now" against "effect then": the difference of two
    # differences, which carries roughly twice the variance of a single arm
    # comparison, hence the sqrt(2) inflation.
    design = sample_size_two_proportions(
        control_rate,
        control_rate + baseline_effect - decayed_effect,
        alpha=alpha,
        power=power,
    )
    n_total = int(np.ceil(design["n_total"] * np.sqrt(2)))
    return {
        "baseline_effect": baseline_effect,
        "detectable_decay": detectable_decay,
        "effect_after_decay": float(decayed_effect),
        "effect_difference_to_detect": float(baseline_effect - decayed_effect),
        "n_total_holdout": n_total,
        "alpha": alpha,
        "power": power,
    }


def duration_table(n_total: int, daily_traffic: list[int]) -> pd.DataFrame:
    """How long the test runs at various traffic levels.

    Included because sample size alone never settles a design argument — the
    question in the room is always "how many weeks", and a test that needs
    fourteen weeks is a different proposal from one that needs two.
    """
    rows = []
    for daily in daily_traffic:
        days = n_total / daily
        rows.append(
            {
                "daily_traffic": daily,
                "days": float(days),
                "weeks": float(days / 7),
                "feasible_in_a_quarter": bool(days <= 90),
            }
        )
    return pd.DataFrame(rows)


def power_curve(
    p_control: float,
    effect: float,
    n_range: np.ndarray,
    alpha: float = 0.05,
    ratio: float = 1.0,
) -> pd.DataFrame:
    """Achieved power across sample sizes, for the design chart."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    p_treated = p_control + effect
    rows = []
    for n_total in n_range:
        n_control = n_total / (1 + ratio)
        se = np.sqrt(
            p_control * (1 - p_control) / n_control + p_treated * (1 - p_treated) / (n_control * ratio)
        )
        power = stats.norm.cdf(abs(effect) / se - z_alpha)
        rows.append({"n_total": int(n_total), "power": float(power)})
    return pd.DataFrame(rows)
