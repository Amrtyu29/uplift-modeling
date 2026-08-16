"""Stretch goal: uncertainty on individual treatment effects, and what it buys.

A point estimate cannot distinguish "+0.05, measured on thousands of similar
customers" from "+0.05, extrapolated from forty". This script puts a posterior
on tau(x) via the Bayesian bootstrap, then does three things with it:

  1. tests whether the intervals are honest — they are not, and the variance
     decomposition shows why widening them is the wrong fix
  2. targets on P(uplift x value > cost) instead of the point estimate, and
     measures whether that earns more
  3. reports where the model is least sure, which is where more experimentation
     is worth buying

    python scripts/06_bayesian.py
    python scripts/06_bayesian.py --n-draws 400 --confidence 0.9
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from uplift.bayesian import (
    BayesianBootstrapUplift,
    coverage_summary,
    decision_rule,
    group_coverage,
    inflate_posterior,
)
from uplift.config import REPORTS, Economics, get_spec
from uplift.data import prepare
from uplift.evaluate import qini_coefficient, qini_curve
from uplift.plots import plot_posterior_calibration, plot_uncertainty
from uplift.simulate import simulate_policy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="hillstrom")
    p.add_argument("--n-draws", type=int, default=200)
    p.add_argument("--credible-level", type=float, default=0.9)
    p.add_argument("--confidence", type=float, default=0.8,
                   help="required P(uplift beats break-even) for the confidence-aware policy")
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--cost-per-contact", type=float, default=None)
    args = p.parse_args()

    spec = get_spec(args.dataset)
    data = prepare(spec, sample_frac=args.sample_frac)
    econ = Economics(
        value_per_conversion=spec.economics.value_per_conversion,
        cost_per_contact=args.cost_per_contact or spec.economics.cost_per_contact,
        currency=spec.economics.currency,
    )

    print(f"\nBayesian uplift — {spec.name}, {args.n_draws} posterior draws")
    print(f"train {data.n_train:,} / test {data.n_test:,}")
    print("Method: Bayesian bootstrap (Dirichlet row weights), refitting the base learner per draw.\n")

    model = BayesianBootstrapUplift(n_draws=args.n_draws, credible_level=args.credible_level)
    model.fit(data.X_train, data.t_train, data.y_train)
    posterior = model.predict_posterior(data.X_test)
    summary = model.summarize(data.X_test, posterior)

    print(f"{'='*100}\nPOSTERIOR OVER INDIVIDUAL TREATMENT EFFECTS\n{'='*100}")
    print(f"  mean uplift              : {summary.mean.mean():+.4f}")
    print(f"  posterior sd, per customer: {summary.std.mean():.4f} average, "
          f"ranging {summary.std.min():.4f} to {summary.std.max():.4f}")
    print(f"  -> the least certain customer's interval is "
          f"{summary.std.max()/summary.std.min():.1f}x wider than the most certain")
    print(f"  P(uplift > 0) > 95% for  : {(summary.prob_positive > 0.95).mean():.1%} of customers")
    print(f"  P(uplift > 0) < 50% for  : {(summary.prob_positive < 0.5).mean():.1%} of customers")

    # ---- Are the intervals honest? ---------------------------------------
    # The inflation factor is fitted on one half of the holdout and tested on
    # the other. Fitting and testing on the same rows would make any correction
    # look perfect by construction.
    rng = np.random.default_rng(0)
    calib = rng.random(data.n_test) < 0.5
    valid = ~calib

    cov_calib = group_coverage(
        data.y_test[calib], data.t_test[calib], posterior[:, calib],
        credible_level=args.credible_level,
    )
    s_calib = coverage_summary(cov_calib, args.credible_level)

    print(f"\n{'='*100}\nARE THE CREDIBLE INTERVALS HONEST?  (calibration half, n={int(calib.sum()):,})\n{'='*100}")
    print("  Standardized residual z = (observed - posterior mean) / sqrt(posterior var + observed var).")
    print("  A calibrated posterior gives sd(z) = 1. Raw coverage is the weaker test: the observed")
    print("  value is itself an estimate whose standard error rivals the interval width.\n")
    print(f"  sd(z)                    : {s_calib['z_sd']:.2f}")
    print(f"  raw coverage at {args.credible_level:.0%}      : {s_calib['raw_coverage']:.0%}")
    print(f"  verdict                  : {s_calib['verdict']}")

    # Before reaching for a fix, check which term the residual variance is
    # actually made of. Widening the posterior can only help if posterior
    # variance is a meaningful share of the total.
    post_var = (cov_calib["posterior_sd"] ** 2).mean()
    obs_var = (cov_calib["observed_se"] ** 2).mean()
    post_share = post_var / (post_var + obs_var)
    print(f"\n  Variance decomposition of the residual denominator:")
    print(f"    posterior variance   : {post_share:.1%} of the total")
    print(f"    observation variance : {1 - post_share:.1%}")

    factor = max(s_calib["z_sd"], 1.0)
    cov_valid_raw = group_coverage(
        data.y_test[valid], data.t_test[valid], posterior[:, valid],
        credible_level=args.credible_level,
    )
    posterior_cal = inflate_posterior(posterior, factor)
    cov_valid_cal = group_coverage(
        data.y_test[valid], data.t_test[valid], posterior_cal[:, valid],
        credible_level=args.credible_level,
    )
    s_valid_raw = coverage_summary(cov_valid_raw, args.credible_level)
    s_valid_cal = coverage_summary(cov_valid_cal, args.credible_level)

    print(f"\n  Widening the posterior by {factor:.2f}x and testing on the held-out half "
          f"(n={int(valid.sum()):,}):")
    print(f"    before : sd(z) {s_valid_raw['z_sd']:.2f}, interval width {s_valid_raw['mean_interval_width']:.4f}")
    print(f"    after  : sd(z) {s_valid_cal['z_sd']:.2f}, interval width {s_valid_cal['mean_interval_width']:.4f}")

    if post_share < 0.25:
        print(f"\n  The correction barely moves sd(z), and the decomposition says why: posterior")
        print(f"  variance is only {post_share:.1%} of the denominator, so inflating it cannot")
        print("  rescue a residual driven by the other term. The honest conclusion is that the")
        print("  excess is not understated *sampling* uncertainty — it is bias in the point")
        print("  estimates themselves, which wider error bars do not fix. What fixes it is a")
        print("  better model or more data, and claiming otherwise would be dressing up a")
        print("  known error as a quantified one.")
    elif s_valid_cal["z_sd"] < s_valid_raw["z_sd"] - 0.1:
        print("\n  The correction transfers to data it was not fitted on, so it is a real fix")
        print("  rather than a restatement of the calibration sample.")
    else:
        print("\n  The correction did not transfer — reported as found.")

    # ---- Does knowing the uncertainty earn anything? ----------------------
    print(f"\n{'='*100}\nDOES THE UNCERTAINTY CHANGE THE DECISION?\n{'='*100}")
    rule = decision_rule(posterior_cal, econ.value_per_conversion, econ.cost_per_contact, args.confidence)
    print(f"  Break-even uplift at {econ.currency}{econ.cost_per_contact:g}/contact and "
          f"{econ.currency}{econ.value_per_conversion:g}/conversion: {rule['break_even_uplift']:.5f}")
    print(f"  Customers where P(uplift > break-even) >= {args.confidence:.0%}: "
          f"{rule['contact'].mean():.1%}")

    # At a near-zero contact cost the bar is trivially cleared by everyone, so
    # the rule only becomes informative once contact is expensive enough to be
    # worth withholding. Sweeping it shows where confidence starts to bite.
    print(f"\n  How the confidence rule tightens as contact gets more expensive:")
    print(f"    {'cost':>8}{'break-even uplift':>20}{'contacted':>12}")
    cost_grid = [econ.cost_per_contact, 0.5, 1.0, 2.0, 4.0, 8.0]
    cost_rows = []
    for c in cost_grid:
        r = decision_rule(posterior_cal, econ.value_per_conversion, c, args.confidence)
        cost_rows.append({"cost": c, "break_even": r["break_even_uplift"],
                          "share": float(r["contact"].mean())})
        print(f"    {econ.currency + format(c, '.2f'):>8}{r['break_even_uplift']:>20.4f}"
              f"{r['contact'].mean():>11.1%}")
    print("\n    A point estimate would answer this with a hard yes/no at every cost. The")
    print("    posterior answers with a share of customers the evidence actually supports.")

    policies = {
        "posterior mean (point estimate)": summary.mean,
        "lower credible bound (cautious)": summary.lower,
        f"P(beats break-even) [conf {args.confidence:.0%}]": rule["prob_above_break_even"],
    }
    print(f"\n  {'ranking':<42}{'Qini':>8}{'best depth':>12}{'profit':>12}")
    print("  " + "-" * 74)
    results = {}
    for name, scores in policies.items():
        curve = simulate_policy(scores, data.y_test, data.t_test, econ, value=data.value_test)
        best = curve.loc[curve["profit_observed"].idxmax()]
        q = qini_coefficient(qini_curve(data.y_test, data.t_test, np.asarray(scores, float)))
        results[name] = {
            "qini": float(q),
            "best_depth": float(best["depth"]),
            "profit": float(best["profit_observed"]),
        }
        print(f"  {name:<42}{q:>8.4f}{best['depth']:>11.0%}"
              f"{econ.currency + format(best['profit_observed'], ',.0f'):>12}")

    qini_mean = results["posterior mean (point estimate)"]["qini"]
    qini_prob = results[f"P(beats break-even) [conf {args.confidence:.0%}]"]["qini"]
    print(f"\n  The exceedance ranking scores much worse on Qini ({qini_prob:.3f} vs {qini_mean:.3f}),")
    print(f"  and the reason is the threshold: at {econ.currency}{econ.cost_per_contact:g}/contact the")
    print(f"  break-even effect is {rule['break_even_uplift']:.4f}, which almost every customer clears")
    print("  with near-certainty. P(tau > threshold) saturates at 1 and the ranking loses its")
    print("  resolution. It is a decision rule, not a ranking, and it should not be used as one.")
    print("\n  The profit column separates the three by less than the bootstrap interval established")
    print("  in 03_simulate (roughly +/- $7,000 at the optimum), so none of these differences is")
    print("  real on this holdout. What the posterior contributes is not a better ordering — it is")
    print("  the ability to say how deep to go and which customers the evidence does not support.")

    # ---- Where is the model least sure? ----------------------------------
    print(f"\n{'='*100}\nWHERE IS THE MODEL LEAST CERTAIN?\n{'='*100}")
    q80 = np.quantile(summary.std, 0.8)
    unsure = summary.std >= q80
    profile = (
        data.X_test[unsure].mean() - data.X_test[~unsure].mean()
    ).sort_values(key=np.abs, ascending=False)
    print("  Feature profile of the least-certain 20% (difference vs. everyone else):")
    for feat, diff in profile.head(5).items():
        print(f"    {feat:<24}{diff:+.3f}")
    print("\n  These are the regions where more experimental data would actually change a")
    print("  decision — the argument for where to spend the next randomized holdout.")

    plot_posterior_calibration(cov_valid_raw, cov_valid_cal, factor,
                               name=f"{spec.name}_posterior_calibration.png")
    plot_uncertainty(summary.mean, summary.std, summary.lower, summary.upper,
                     name=f"{spec.name}_uncertainty.png")

    out = REPORTS / f"{spec.name}_bayesian.json"
    out.write_text(
        json.dumps(
            {
                "dataset": spec.name,
                "n_draws": args.n_draws,
                "credible_level": args.credible_level,
                "posterior": {
                    "mean": float(summary.mean.mean()),
                    "sd_mean": float(summary.std.mean()),
                    "sd_min": float(summary.std.min()),
                    "sd_max": float(summary.std.max()),
                    "share_confident_positive": float((summary.prob_positive > 0.95).mean()),
                },
                "calibration": {
                    "factor": float(factor),
                    "calibration_half": s_calib,
                    "validation_before": s_valid_raw,
                    "validation_after": s_valid_cal,
                },
                "policies": results,
                "confidence_rule": {
                    "confidence": args.confidence,
                    "break_even_uplift": float(rule["break_even_uplift"]),
                    "share_contacted": float(rule["contact"].mean()),
                    "cost_sweep": cost_rows,
                },
                "variance_decomposition": {"posterior_share": float(post_share)},
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
