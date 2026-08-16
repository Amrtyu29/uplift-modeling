"""Uplift evaluation: Qini and uplift curves, AUUC, decile tables.

Why not AUC/accuracy: the quantity being predicted — the difference between two
potential outcomes — is never observed for any individual. There is no label to
score against. What *is* available is the randomized holdout, where treated and
control units with similar predicted uplift are exchangeable by construction, so
their observed outcome difference is an unbiased estimate of the true effect for
that group. Every metric here is built from that one idea.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE


def _order(uplift: np.ndarray, random_state: int | None = None) -> np.ndarray:
    """Rank units by predicted uplift, descending, breaking ties at random.

    Tie-breaking matters more than it looks: an S-learner that collapses to a
    constant produces one giant tie, and a stable sort would silently score it
    against whatever order the rows happened to arrive in.
    """
    rng = np.random.default_rng(random_state)
    return np.lexsort((rng.random(len(uplift)), -uplift))


def qini_curve(
    y: np.ndarray, t: np.ndarray, uplift: np.ndarray, random_state: int | None = RANDOM_STATE
) -> pd.DataFrame:
    """Cumulative incremental outcomes as the targeted fraction grows.

    At each cut-off k (units ranked by predicted uplift):

        qini(k) = Y_T(k) - Y_C(k) * N_T(k) / N_C(k)

    i.e. outcomes among targeted treated units, minus what the control group in
    the same slice would have produced had it been the same size. The rescaling
    is what makes this valid off a 50/50 split.

    Also returns the ``uplift`` curve — the same idea expressed as
    (rate_T - rate_C) * n — which is the one to read for "how many extra
    conversions at this depth", and the random-targeting diagonal.
    """
    y, t, uplift = np.asarray(y, float), np.asarray(t, int), np.asarray(uplift, float)
    idx = _order(uplift, random_state)
    y, t = y[idx], t[idx]

    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    y_t = np.cumsum(y * t)
    y_c = np.cumsum(y * (1 - t))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(n_c > 0, n_t / np.maximum(n_c, 1), 0.0)
        qini = y_t - y_c * ratio
        rate_t = np.where(n_t > 0, y_t / np.maximum(n_t, 1), 0.0)
        rate_c = np.where(n_c > 0, y_c / np.maximum(n_c, 1), 0.0)

    n = np.arange(1, len(y) + 1)
    lift = (rate_t - rate_c) * n
    # A slice with only one arm in it cannot support an estimate.
    valid = (n_t > 0) & (n_c > 0)
    qini = np.where(valid, qini, 0.0)
    lift = np.where(valid, lift, 0.0)

    total = qini[-1]
    return pd.DataFrame(
        {
            "n_targeted": n,
            "fraction_targeted": n / len(y),
            "qini": qini,
            "uplift": lift,
            "random": total * n / len(y),
            "rate_treated": rate_t,
            "rate_control": rate_c,
            # Carried so the coefficient can tell "no effect to find" apart from
            # "failed to find the effect" without needing y and t again.
            "n_treated": n_t,
            "n_control": n_c,
        }
    )


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Normalized area between the Qini curve and the random-targeting line.

    The raw area is in units of (conversions x customers), which is impossible
    to compare across datasets. Dividing by the area a *perfect* ranking would
    sweep — one that front-loads every incremental conversion, giving a curve
    that jumps to the total immediately, i.e. area ``Q_total * n / 2`` above the
    diagonal — puts this on a 0-to-1 scale, exactly analogous to a Gini
    coefficient.

    0 means the ranking is no better than random. Negative means it is worse
    than random, which is a real and reportable outcome, not a bug. Values
    around 0.05-0.15 are normal on real marketing data; treatment effects are
    genuinely hard to rank.

    It can exceed 1, and that is not a bug either. The denominator is the effect
    available from treating *everyone*, so when a meaningful share of the
    population responds negatively, a ranking that targets only the positive
    responders accumulates more incremental outcome than the whole population
    delivers — the curve rises above its own endpoint. Normalizing by the truly
    optimal curve instead would need the individual treatment effects, which is
    precisely what is unobservable.

    Returns NaN when the campaign has no overall effect to allocate. As that
    same denominator approaches zero the ratio diverges, so an experiment where
    nothing happened can score 1.2 on pure noise and read as an excellent model.
    Since the metric asks "what share of the achievable effect did this ranking
    capture", it is undefined — not large — when nothing is achievable. The test
    is against the endpoint's own standard error, not against exact zero.
    """
    x = curve["n_targeted"].to_numpy(float)
    area = float(np.trapezoid(curve["qini"] - curve["random"], x))
    total, n = float(curve["qini"].iloc[-1]), len(curve)

    n_t, n_c = float(curve["n_treated"].iloc[-1]), float(curve["n_control"].iloc[-1])
    p1, p0 = float(curve["rate_treated"].iloc[-1]), float(curve["rate_control"].iloc[-1])
    if n_t == 0 or n_c == 0:
        return float("nan")
    # SE of the total incremental count = n_t * SE(p1 - p0), matching the scale
    # of `total`, which is expressed in treated-equivalent conversions.
    se_total = n_t * np.sqrt(p1 * (1 - p1) / n_t + p0 * (1 - p0) / n_c)
    if total <= 1.96 * se_total:
        return float("nan")

    denom = total * n / 2
    return area / denom if denom != 0 else float("nan")


def auuc(curve: pd.DataFrame, normalize: bool = True) -> float:
    """Area under the uplift curve.

    Normalized by (n_units * overall_ATE * n_units / 2) — the area a perfectly
    flat-effect model would sweep — so values are comparable across datasets
    with different base rates. Unnormalized it is in units of conversions.
    """
    x = curve["n_targeted"].to_numpy(float)
    area = float(np.trapezoid(curve["uplift"], x))
    if not normalize:
        return area
    n = len(curve)
    overall = curve["uplift"].iloc[-1]
    denom = overall * n / 2
    return area / denom if denom != 0 else float("nan")


def qini_with_ci(
    y: np.ndarray,
    t: np.ndarray,
    uplift: np.ndarray,
    n_boot: int = 200,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Bootstrap CI for the Qini coefficient.

    Uplift metrics are noisy — the estimate is a difference of two rates inside
    every slice — and a point estimate on its own invites over-reading a model
    ranking. If two models' intervals overlap heavily, they are not
    distinguishable on this holdout, and the write-up says so.
    """
    rng = np.random.default_rng(random_state)
    n = len(y)
    point = qini_coefficient(qini_curve(y, t, uplift, random_state))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        # A resample can lose an arm entirely in tiny datasets; skip those draws.
        if len(np.unique(t[idx])) < 2:
            boots[b] = np.nan
            continue
        boots[b] = qini_coefficient(qini_curve(y[idx], t[idx], uplift[idx], random_state + b))
    # Draws are dropped when a resample loses an arm, and now also when the
    # coefficient itself is undefined because the resampled effect is
    # indistinguishable from zero. Both can empty the array, so the interval
    # degrades to NaN rather than raising out of a percentile call.
    boots = boots[~np.isnan(boots)]
    if len(boots) < 2:
        return {
            "qini_coefficient": point,
            "qini_ci_low": float("nan"),
            "qini_ci_high": float("nan"),
            "qini_boot_std": float("nan"),
            "n_boot_valid": int(len(boots)),
        }
    return {
        "qini_coefficient": point,
        "qini_ci_low": float(np.percentile(boots, 2.5)),
        "qini_ci_high": float(np.percentile(boots, 97.5)),
        "qini_boot_std": float(boots.std(ddof=1)),
        "n_boot_valid": int(len(boots)),
    }


def decile_table(y: np.ndarray, t: np.ndarray, uplift: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Observed uplift per predicted-uplift bin — the model's honesty check.

    A model that ranks well produces a monotone decreasing column here. This is
    also where sleeping dogs show up: a bottom decile whose observed uplift is
    negative with a CI excluding zero is a group the intervention is hurting.
    """
    y, t, uplift = np.asarray(y, float), np.asarray(t, int), np.asarray(uplift, float)
    order = _order(uplift)
    ranks = np.empty(len(uplift), int)
    ranks[order] = np.arange(len(uplift))
    group = np.minimum((ranks * bins) // len(uplift), bins - 1)

    rows = []
    for g in range(bins):
        m = group == g
        yt, yc = y[m & (t == 1)], y[m & (t == 0)]
        p1 = yt.mean() if len(yt) else np.nan
        p0 = yc.mean() if len(yc) else np.nan
        se = np.sqrt(
            (p1 * (1 - p1) / len(yt) if len(yt) else np.nan)
            + (p0 * (1 - p0) / len(yc) if len(yc) else np.nan)
        )
        rows.append(
            {
                "decile": g + 1,
                "n": int(m.sum()),
                "n_treated": len(yt),
                "n_control": len(yc),
                "predicted_uplift": float(uplift[m].mean()),
                "observed_uplift": float(p1 - p0),
                "observed_ci_low": float(p1 - p0 - 1.96 * se),
                "observed_ci_high": float(p1 - p0 + 1.96 * se),
                "rate_treated": float(p1),
                "rate_control": float(p0),
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    y: np.ndarray,
    t: np.ndarray,
    uplift: np.ndarray,
    name: str = "model",
    n_boot: int = 200,
) -> dict:
    """Full evaluation bundle for one model on one holdout."""
    curve = qini_curve(y, t, uplift)
    metrics = {
        "model": name,
        "auuc": auuc(curve),
        "auuc_raw": auuc(curve, normalize=False),
        "uplift_at_20pct": float(curve.loc[int(0.2 * len(curve)), "uplift"]),
        "uplift_at_30pct": float(curve.loc[int(0.3 * len(curve)), "uplift"]),
        "total_incremental": float(curve["uplift"].iloc[-1]),
        "pred_mean": float(uplift.mean()),
        "pred_std": float(uplift.std()),
        "pred_negative_share": float((uplift < 0).mean()),
    }
    metrics.update(qini_with_ci(y, t, uplift, n_boot=n_boot))
    return {"metrics": metrics, "curve": curve, "deciles": decile_table(y, t, uplift)}
