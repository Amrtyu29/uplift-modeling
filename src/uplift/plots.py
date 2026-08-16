"""Report figures. Matplotlib only, so they render identically in CI and locally."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES
from .segments import SEGMENT_ORDER

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)

PALETTE = {
    "uplift": "#1b6ca8",
    "response": "#e07b39",
    # Policy frames name this column "response_model"; without the alias the
    # lookup misses, matplotlib falls back to its default blue, and the two
    # policies become indistinguishable on the chart.
    "response_model": "#e07b39",
    "random": "#9aa0a6",
    "treat_all": "#5a5a5a",
    "s_learner": "#8e6fb5",
    "t_learner": "#e07b39",
    "x_learner": "#1b6ca8",
    "class_transform": "#3f9e5a",
    "causal_forest": "#c0392b",
}


def save_figure(fig, name: str, outdir: Path | None = None) -> Path:
    outdir = outdir or FIGURES
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_qini_curves(curves: dict[str, pd.DataFrame], title: str, outdir=None, name="qini_curves.png"):
    """Qini curves for every model against the random-targeting diagonal."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ref = next(iter(curves.values()))
    ax.plot(
        ref["fraction_targeted"] * 100,
        ref["random"],
        "--",
        color=PALETTE["random"],
        label="random targeting",
        lw=1.5,
    )
    for model, curve in curves.items():
        ax.plot(
            curve["fraction_targeted"] * 100,
            curve["qini"],
            label=model.replace("_", "-"),
            color=PALETTE.get(model),
            lw=2,
        )
    ax.set_xlabel("% of population targeted (ranked by predicted uplift)")
    ax.set_ylabel("Cumulative incremental conversions")
    ax.set_title(title)
    ax.legend(frameon=False)
    return save_figure(fig, name, outdir)


def plot_money_chart(
    comparison: pd.DataFrame,
    economics,
    profit_col: str = "profit_observed",
    title: str = "Incremental profit vs. share of customers targeted",
    outdir=None,
    name="money_chart.png",
    band: pd.DataFrame | None = None,
):
    """The headline chart: profit as a function of targeting depth, per policy.

    ``band`` adds the bootstrap CI around the uplift policy. Without it the
    curve reads as far more precise than a 19K-row holdout can support.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    if band is not None and f"{profit_col}_lo" in band:
        ax.fill_between(
            band["depth"] * 100, band[f"{profit_col}_lo"], band[f"{profit_col}_hi"],
            color=PALETTE["uplift"], alpha=0.15, label="uplift policy, 95% CI",
        )
    for policy, grp in comparison.groupby("policy"):
        grp = grp.sort_values("depth")
        style = "--" if policy in ("random", "treat_all") else "-"
        ax.plot(
            grp["depth"] * 100,
            grp[profit_col],
            style,
            label=policy.replace("_", " "),
            color=PALETTE.get(policy),
            lw=2.2 if style == "-" else 1.6,
        )
    up = comparison[comparison["policy"] == "uplift"].sort_values("depth")
    if len(up):
        best = up.loc[up[profit_col].idxmax()]
        ax.scatter([best["depth"] * 100], [best[profit_col]], zorder=5, color=PALETTE["uplift"], s=60)
        # Anchor right of the point unless the optimum sits near the right edge,
        # where the label would otherwise run off the axes.
        right_edge = best["depth"] > 0.75
        ax.annotate(
            f"optimum: top {best['depth']*100:.0f}%\n{economics.currency}{best[profit_col]:,.0f}",
            (best["depth"] * 100, best[profit_col]),
            textcoords="offset points",
            xytext=(-12, 14) if right_edge else (12, -6),
            ha="right" if right_edge else "left",
            fontsize=9,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("% of customers contacted")
    ax.set_ylabel(f"Incremental profit ({economics.currency}, holdout)")
    ax.set_title(title)
    ax.legend(frameon=False)
    return save_figure(fig, name, outdir)


def plot_break_even(
    sweep: pd.DataFrame,
    economics,
    current_cost: float | None = None,
    outdir=None,
    name="break_even.png",
):
    """When the uplift policy is worth running, as a function of contact cost.

    The top panel is the decision: profit under selective targeting vs.
    contacting everyone. The bottom panel shows how deep the optimal policy
    goes — it narrows as contact gets expensive, which is the mechanism behind
    the gain.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True)
    c = sweep["cost_per_contact"]
    axes[0].plot(c, sweep["profit_uplift"], color=PALETTE["uplift"], lw=2.2, label="uplift policy (best depth)")
    axes[0].plot(c, sweep["profit_treat_all"], "--", color=PALETTE["treat_all"], lw=1.8, label="treat everyone")
    axes[0].axhline(0, color="black", lw=0.8)

    wins = sweep[sweep["uplift_wins"]]
    if len(wins):
        be = wins["cost_per_contact"].min()
        axes[0].axvline(be, color=PALETTE["response"], ls=":", lw=1.8)
        axes[0].annotate(
            f"break-even\n{economics.currency}{be:.2f}/contact",
            (be, axes[0].get_ylim()[1] * 0.75),
            textcoords="offset points", xytext=(8, 0), fontsize=9, color=PALETTE["response"],
        )
    if current_cost is not None:
        axes[0].axvline(current_cost, color="#666", ls="-.", lw=1.4)
        axes[0].annotate(
            f"assumed\n{economics.currency}{current_cost:g}",
            (current_cost, axes[0].get_ylim()[1] * 0.3),
            textcoords="offset points", xytext=(8, 0), fontsize=9, color="#666",
        )
    axes[0].set_ylabel(f"Profit ({economics.currency})")
    axes[0].set_title("Does selective targeting beat contacting everyone?")
    axes[0].legend(frameon=False)

    axes[1].plot(c, sweep["best_depth"] * 100, color=PALETTE["uplift"], lw=2)
    axes[1].set_ylabel("Optimal depth (% targeted)")
    axes[1].set_xlabel(f"Cost per contact ({economics.currency})")
    axes[1].set_ylim(0, 105)
    return save_figure(fig, name, outdir)


def plot_decile_calibration(deciles: pd.DataFrame, title: str, outdir=None, name="deciles.png"):
    """Predicted vs. observed uplift per decile, with CIs on the observed side."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = deciles["decile"]
    err = [
        deciles["observed_uplift"] - deciles["observed_ci_low"],
        deciles["observed_ci_high"] - deciles["observed_uplift"],
    ]
    ax.bar(x, deciles["observed_uplift"], color=PALETTE["uplift"], alpha=0.75, label="observed")
    ax.errorbar(x, deciles["observed_uplift"], yerr=err, fmt="none", ecolor="#333", lw=1, capsize=3)
    ax.plot(x, deciles["predicted_uplift"], "o-", color=PALETTE["response"], label="predicted", lw=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Decile of predicted uplift (1 = highest)")
    ax.set_ylabel("Uplift (treated rate − control rate)")
    ax.set_title(title)
    ax.set_xticks(list(x))
    ax.legend(frameon=False)
    return save_figure(fig, name, outdir)


def plot_segments(summary: pd.DataFrame, title: str, outdir=None, name="segments.png"):
    """Segment sizes and their observed uplift, side by side."""
    summary = summary.set_index("segment").reindex([s for s in SEGMENT_ORDER if s in set(summary["segment"])])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = ["#1b6ca8", "#c0392b", "#3f9e5a", "#9aa0a6"]
    labels = [s.replace("_", " ") for s in summary.index]

    axes[0].bar(labels, summary["share"] * 100, color=colors[: len(summary)])
    axes[0].set_ylabel("% of customers")
    axes[0].set_title("Segment sizes")

    err = [
        summary["observed_uplift"] - summary["ci_low"],
        summary["ci_high"] - summary["observed_uplift"],
    ]
    axes[1].bar(labels, summary["observed_uplift"], color=colors[: len(summary)])
    axes[1].errorbar(
        labels, summary["observed_uplift"], yerr=err, fmt="none", ecolor="#333", lw=1, capsize=4
    )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Observed uplift on holdout")
    axes[1].set_title("Observed uplift by segment (95% CI)")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle(title, fontweight="bold")
    return save_figure(fig, name, outdir)


def plot_uplift_distribution(uplift: np.ndarray, title: str, outdir=None, name="uplift_distribution.png"):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(uplift, bins=60, color=PALETTE["uplift"], alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(float(np.mean(uplift)), color=PALETTE["response"], ls="--", lw=1.5, label="mean (ATE estimate)")
    ax.set_xlabel("Predicted uplift")
    ax.set_ylabel("Customers")
    ax.set_title(title)
    ax.legend(frameon=False)
    return save_figure(fig, name, outdir)


def plot_naive_vs_uplift(
    comparison: pd.DataFrame,
    economics,
    outdir=None,
    name="naive_vs_uplift.png",
):
    """The talking-point chart: a response model targets the wrong people.

    Both curves rank the same customers on the same holdout — the only
    difference is whether the ranking is 'most likely to convert' or 'most
    moved by the offer'.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for policy in ("uplift", "response_model", "random"):
        grp = comparison[comparison["policy"] == policy].sort_values("depth")
        if not len(grp):
            continue
        ax.plot(
            grp["depth"] * 100,
            grp["incremental_conversions"],
            "--" if policy == "random" else "-",
            label={"uplift": "uplift ranking", "response_model": "response-probability ranking", "random": "random"}[policy],
            color=PALETTE.get({"response_model": "response"}.get(policy, policy)),
            lw=2.2,
        )
    ax.set_xlabel("% of customers contacted")
    ax.set_ylabel("Incremental conversions (holdout)")
    ax.set_title("Uplift ranking vs. a conventional response model")
    ax.legend(frameon=False)
    return save_figure(fig, name, outdir)


def plot_drift(history: pd.DataFrame, outdir=None, name="drift_monitor.png"):
    """Monitoring view: mean predicted uplift and PSI per batch, with thresholds."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(history["batch"], history["mean_uplift"], "o-", color=PALETTE["uplift"])
    axes[0].fill_between(
        history["batch"], history["ref_mean_low"], history["ref_mean_high"], color=PALETTE["uplift"], alpha=0.15,
        label="reference range",
    )
    axes[0].set_ylabel("Mean predicted uplift")
    axes[0].set_title("Treatment-effect drift monitor")
    axes[0].legend(frameon=False)

    axes[1].plot(history["batch"], history["psi"], "o-", color=PALETTE["response"])
    axes[1].axhline(0.1, ls="--", color="#999", label="PSI 0.10 (watch)")
    axes[1].axhline(0.25, ls="--", color="#c0392b", label="PSI 0.25 (alert)")
    axes[1].set_ylabel("PSI vs. reference")
    axes[1].set_xlabel("Batch")
    axes[1].legend(frameon=False)
    return save_figure(fig, name, outdir)


def plot_posterior_calibration(
    before: pd.DataFrame,
    after: pd.DataFrame,
    factor: float,
    outdir=None,
    name="posterior_calibration.png",
):
    """Credible intervals against observed effects, before and after widening.

    Both panels show the same held-out bins; only the stated confidence changes.
    Error bars on the observed points are their own standard errors, which is
    the reason raw coverage is a misleading test on its own — the "truth" here
    is itself an estimate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, df, label in ((axes[0], before, "as estimated"), (axes[1], after, f"widened {factor:.2f}x")):
        x = df["bin"]
        ax.errorbar(
            x, df["posterior_mean"],
            yerr=[df["posterior_mean"] - df["credible_low"], df["credible_high"] - df["posterior_mean"]],
            fmt="o", color=PALETTE["uplift"], capsize=4, lw=2, label="posterior (credible interval)",
        )
        ax.errorbar(
            x, df["observed_uplift"], yerr=1.96 * df["observed_se"],
            fmt="s", color=PALETTE["response"], capsize=3, lw=1.4, alpha=0.9,
            label="observed (95% CI)",
        )
        sd = df["z"].std(ddof=1)
        ax.set_title(f"{label} — sd(z) = {sd:.2f}")
        ax.set_xlabel("Bin of predicted uplift (1 = highest)")
        ax.set_xticks(list(x))
    axes[0].set_ylabel("Uplift")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Are the credible intervals honest?", fontweight="bold")
    return save_figure(fig, name, outdir)


def plot_uncertainty(mean, std, lower, upper, outdir=None, name="uncertainty.png"):
    """Per-customer uncertainty: how it varies, and how it scales with the estimate."""
    order = np.argsort(-np.asarray(mean))
    m, lo, hi = np.asarray(mean)[order], np.asarray(lower)[order], np.asarray(upper)[order]
    x = np.arange(len(m))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    # Thinned for legibility: 19K overlapping bands render as a solid block.
    step = max(1, len(m) // 2000)
    axes[0].fill_between(x[::step], lo[::step], hi[::step], color=PALETTE["uplift"], alpha=0.25,
                         label="credible interval")
    axes[0].plot(x[::step], m[::step], color=PALETTE["uplift"], lw=1.6, label="posterior mean")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xlabel("Customers, ranked by predicted uplift")
    axes[0].set_ylabel("Uplift")
    axes[0].set_title("Effect and its uncertainty, per customer")
    axes[0].legend(frameon=False)

    axes[1].scatter(mean, std, s=4, alpha=0.25, color=PALETTE["uplift"])
    axes[1].set_xlabel("Posterior mean uplift")
    axes[1].set_ylabel("Posterior sd")
    axes[1].set_title("Where the model is least sure")
    return save_figure(fig, name, outdir)


def plot_power_analysis(
    curve: pd.DataFrame,
    n_required: int,
    target_power: float = 0.8,
    outdir=None,
    name="power_analysis.png",
):
    """Power against sample size for the policy comparison.

    Log x-axis on purpose: the required sample size for a policy test is often
    an order of magnitude above the treatment test, and a linear axis hides
    exactly the region where the decision is made.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(curve["n_total"], curve["power"], color=PALETTE["uplift"], lw=2.4)
    ax.axhline(target_power, ls="--", color=PALETTE["response"], lw=1.5,
               label=f"target power {target_power:.0%}")
    ax.axvline(n_required, ls=":", color="#666", lw=1.5,
               label=f"required n = {n_required:,}")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Total sample size (both arms, log scale)")
    ax.set_ylabel("Power to detect the policy difference")
    ax.set_title("Sizing the test of uplift targeting vs. blanket targeting")
    ax.legend(frameon=False)
    return save_figure(fig, name, outdir)
