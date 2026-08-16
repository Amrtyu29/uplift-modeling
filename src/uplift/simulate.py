"""Business simulation: what a targeting policy would have earned.

The simulation never invents outcomes. It takes a policy — a ranking plus a
depth — applies it to the randomized holdout, and reads the incremental effect
straight off the treated/control gap *within the targeted slice*. That gap is
unbiased because assignment was random and the ranking is a function of
pre-treatment features only, so it cannot have peeked at the outcome.

What is assumed rather than measured: the unit economics (value per conversion,
cost per contact), and that the observed campaign effect would hold at rollout.
Both are stated in every report this module produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RANDOM_STATE, Economics
from .evaluate import _order


def _slice_effect(
    mask: np.ndarray, y: np.ndarray, t: np.ndarray, value: np.ndarray | None
) -> dict[str, float]:
    """Observed incremental outcome and revenue inside a targeted slice."""
    yt, yc = y[mask & (t == 1)], y[mask & (t == 0)]
    n = int(mask.sum())
    if len(yt) == 0 or len(yc) == 0:
        return {"incremental_conversions": 0.0, "incremental_value": 0.0, "n_targeted": n}
    rate_diff = yt.mean() - yc.mean()
    out = {"incremental_conversions": float(rate_diff * n), "n_targeted": n}
    if value is not None:
        vt, vc = value[mask & (t == 1)], value[mask & (t == 0)]
        out["incremental_value"] = float((vt.mean() - vc.mean()) * n)
    else:
        out["incremental_value"] = float("nan")
    return out


def simulate_policy(
    scores: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    economics: Economics,
    value: np.ndarray | None = None,
    depths: np.ndarray | None = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Profit curve for targeting the top-k fraction by ``scores``.

    ``incremental_value`` uses the dataset's own observed spend where available
    and is the number to trust. ``profit_modeled`` instead prices conversions at
    the assumed ``value_per_conversion`` — useful when the dataset has no
    revenue column, and useful as a cross-check when it does.
    """
    y, t = np.asarray(y, float), np.asarray(t, int)
    value = None if value is None else np.asarray(value, float)
    n = len(y)
    depths = np.arange(0.05, 1.0001, 0.05) if depths is None else np.asarray(depths)

    order = _order(np.asarray(scores, float), random_state)
    rows = []
    for d in depths:
        k = max(1, int(round(d * n)))
        mask = np.zeros(n, bool)
        mask[order[:k]] = True
        eff = _slice_effect(mask, y, t, value)
        cost = k * economics.cost_per_contact
        rows.append(
            {
                "depth": float(d),
                "n_targeted": k,
                "incremental_conversions": eff["incremental_conversions"],
                "incremental_value": eff["incremental_value"],
                "cost": cost,
                "profit_observed": eff["incremental_value"] - cost,
                "profit_modeled": eff["incremental_conversions"] * economics.value_per_conversion
                - cost,
            }
        )
    df = pd.DataFrame(rows)
    df["roi"] = np.where(df["cost"] > 0, df["profit_observed"] / df["cost"], np.nan)
    return df


def simulate_random(
    y: np.ndarray,
    t: np.ndarray,
    economics: Economics,
    value: np.ndarray | None = None,
    depths: np.ndarray | None = None,
    n_draws: int = 20,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Random-targeting baseline, averaged over draws.

    Averaging matters: a single random ranking on a 19K-row holdout is noisy
    enough to occasionally beat a real model at some depths, which would make
    the comparison look flattering or damning at random.
    """
    rng = np.random.default_rng(random_state)
    frames = [
        simulate_policy(
            rng.random(len(y)), y, t, economics, value, depths, random_state=random_state + i
        )
        for i in range(n_draws)
    ]
    out = pd.concat(frames).groupby("depth", as_index=False).mean(numeric_only=True)
    return out


def compare_policies(
    policies: dict[str, np.ndarray],
    y: np.ndarray,
    t: np.ndarray,
    economics: Economics,
    value: np.ndarray | None = None,
    depths: np.ndarray | None = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Long-format profit curves for every policy plus random and treat-all."""
    frames = []
    for name, scores in policies.items():
        f = simulate_policy(scores, y, t, economics, value, depths, random_state)
        f.insert(0, "policy", name)
        frames.append(f)

    rand = simulate_random(y, t, economics, value, depths, random_state=random_state)
    rand.insert(0, "policy", "random")
    frames.append(rand)

    # Treat-everyone is the incumbent most campaigns actually run: it is the
    # 100%-depth point of any policy, held flat across the axis for comparison.
    full = simulate_policy(np.zeros(len(y)), y, t, economics, value, [1.0], random_state)
    ref_depths = frames[0]["depth"].to_numpy()
    treat_all = pd.DataFrame(
        {
            "policy": "treat_all",
            "depth": ref_depths,
            "n_targeted": len(y),
            "incremental_conversions": full["incremental_conversions"].iloc[0],
            "incremental_value": full["incremental_value"].iloc[0],
            "cost": full["cost"].iloc[0],
            "profit_observed": full["profit_observed"].iloc[0],
            "profit_modeled": full["profit_modeled"].iloc[0],
            "roi": full["roi"].iloc[0],
        }
    )
    frames.append(treat_all)
    return pd.concat(frames, ignore_index=True)


def bootstrap_profit_curve(
    scores: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    economics: Economics,
    value: np.ndarray | None = None,
    depths: np.ndarray | None = None,
    n_boot: int = 200,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Percentile CI band around a policy's profit curve.

    Worth the compute: the point estimate at each depth is a difference of two
    heavy-tailed revenue means inside a slice, so the raw curve wobbles enough
    to look like structure. Plotting the band makes clear which wiggles are
    signal and which are the holdout being 19K rows.
    """
    rng = np.random.default_rng(random_state)
    scores, y, t = np.asarray(scores), np.asarray(y), np.asarray(t)
    n = len(y)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(t[idx])) < 2:
            continue
        draws.append(
            simulate_policy(
                scores[idx], y[idx], t[idx], economics,
                None if value is None else np.asarray(value)[idx],
                depths, random_state=random_state,
            ).set_index("depth")
        )
    stacked = {col: np.vstack([d[col].to_numpy() for d in draws]) for col in
               ("profit_observed", "profit_modeled", "incremental_conversions", "incremental_value")}
    out = pd.DataFrame({"depth": draws[0].index.to_numpy()})
    for col, arr in stacked.items():
        out[f"{col}_lo"] = np.percentile(arr, 2.5, axis=0)
        out[f"{col}_hi"] = np.percentile(arr, 97.5, axis=0)
    return out


def break_even_contact_cost(
    scores: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    economics: Economics,
    value: np.ndarray | None = None,
    depths: np.ndarray | None = None,
    costs: np.ndarray | None = None,
    basis: str = "observed",
    n_boot: int = 200,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """How the optimal policy changes as contact gets more expensive.

    This is the honest answer to "is the uplift model worth anything here?" —
    it depends entirely on what a contact costs. Email at a tenth of a cent is
    so cheap that contacting everyone is close to optimal no matter how good
    the ranking is; the model only earns money once cost per contact approaches
    the incremental revenue per contact. Rather than pick a flattering cost
    assumption, sweep it and report the break-even point.

    The incremental-value curve does not depend on cost, so it is computed once
    and re-priced for every cost in the sweep.

    "Wins" is a statistical claim, not a point comparison: the gain over
    treat-everyone is bootstrapped and the policy only counts as winning when
    the lower bound clears zero. Without that guard the sweep reports a
    break-even of zero, because at zero cost some slice of the holdout always
    happens to out-earn the full population by a few dollars of noise — when in
    truth, with every effect positive and contact free, contacting everyone
    cannot be beaten.

    ``basis`` picks what revenue means: ``observed`` uses the dataset's revenue
    column, ``modeled`` prices incremental conversions at
    ``economics.value_per_conversion``. Observed is truer but much noisier,
    since it differences a heavy-tailed spend variable inside every slice.
    """
    value_key = "incremental_value" if basis == "observed" else "incremental_conversions"
    scale = 1.0 if basis == "observed" else economics.value_per_conversion

    base = simulate_policy(scores, y, t, economics, value, depths, random_state)
    n_total = len(y)
    n_targeted = base["n_targeted"].to_numpy()
    rev = base[value_key].to_numpy() * scale
    rev_all = rev[-1]

    if costs is None:
        # Span from free to well past the incremental revenue per contact,
        # which is where every policy stops being profitable at all.
        rev_per_contact = rev_all / n_total if rev_all > 0 else 1.0
        costs = np.linspace(0, max(rev_per_contact * 2, 1e-6), 41)

    # Bootstrap the revenue curves once; re-price them at every cost.
    rng = np.random.default_rng(random_state)
    scores_a, y_a, t_a = np.asarray(scores), np.asarray(y), np.asarray(t)
    value_a = None if value is None else np.asarray(value)
    boot_rev = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_total, n_total)
        if len(np.unique(t_a[idx])) < 2:
            continue
        b = simulate_policy(
            scores_a[idx], y_a[idx], t_a[idx], economics,
            None if value_a is None else value_a[idx], depths, random_state=random_state,
        )
        boot_rev.append(b[value_key].to_numpy() * scale)
    boot_rev = np.vstack(boot_rev)

    rows = []
    for c in costs:
        profit_u = rev - c * n_targeted
        profit_all = rev_all - c * n_total
        # Depth is chosen on the point estimate, then that fixed depth is
        # evaluated across bootstrap draws. Re-optimizing inside each draw
        # would bake in the winner's curse and overstate the gain.
        k = int(np.argmax(profit_u))
        gains = (boot_rev[:, k] - c * n_targeted[k]) - (boot_rev[:, -1] - c * n_total)
        lo, hi = np.percentile(gains, [2.5, 97.5])
        rows.append(
            {
                "cost_per_contact": float(c),
                "best_depth": float(base["depth"].iloc[k]),
                "profit_uplift": float(profit_u[k]),
                "profit_treat_all": float(profit_all),
                "gain_vs_treat_all": float(profit_u[k] - profit_all),
                "gain_lo": float(lo),
                "gain_hi": float(hi),
                "gain_pct": float((profit_u[k] - profit_all) / abs(profit_all) * 100) if profit_all else np.nan,
                "incremental_conversions": float(base["incremental_conversions"].iloc[k]),
            }
        )
    df = pd.DataFrame(rows)
    df["uplift_wins"] = df["gain_lo"] > 0
    return df


def optimal_depth(curve: pd.DataFrame, profit_col: str = "profit_observed") -> dict:
    """Where a policy's profit curve peaks, and how it compares to treating all."""
    best = curve.loc[curve[profit_col].idxmax()]
    return {
        "best_depth": float(best["depth"]),
        "n_targeted": int(best["n_targeted"]),
        "profit": float(best[profit_col]),
        "incremental_conversions": float(best["incremental_conversions"]),
        "incremental_value": float(best["incremental_value"]),
        "cost": float(best["cost"]),
    }


def headline(
    comparison: pd.DataFrame, profit_col: str = "profit_observed", policy: str = "uplift"
) -> dict:
    """The two numbers that belong in the README and the resume bullet."""
    uplift_curve = comparison[comparison["policy"] == policy]
    best = optimal_depth(uplift_curve, profit_col)
    all_profit = float(comparison[comparison["policy"] == "treat_all"][profit_col].iloc[0])
    rand_at_depth = comparison[
        (comparison["policy"] == "random")
        & np.isclose(comparison["depth"], best["best_depth"])
    ][profit_col]
    rand_profit = float(rand_at_depth.iloc[0]) if len(rand_at_depth) else float("nan")

    def _gain(base: float) -> float:
        if not np.isfinite(base) or base == 0:
            return float("nan")
        return (best["profit"] - base) / abs(base) * 100

    return {
        **best,
        "profit_treat_all": all_profit,
        "profit_random_same_depth": rand_profit,
        "gain_vs_treat_all_pct": _gain(all_profit),
        "gain_vs_random_pct": _gain(rand_profit),
        "contacts_saved_vs_treat_all": 1 - best["best_depth"],
    }
