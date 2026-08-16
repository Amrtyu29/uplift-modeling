"""Days 3-6: fit every learner, score them on the randomized holdout, pick one.

    python scripts/02_train.py --dataset hillstrom
    python scripts/02_train.py --dataset criteo --sample-frac 0.15

Writes: models/<dataset>_<learner>.joblib, reports/<dataset>_metrics.json,
reports/figures/*.png
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from uplift.config import FIGURES, REPORTS, get_spec
from uplift.data import prepare
from uplift.evaluate import decile_table
from uplift.pipeline import (
    build_bundle,
    cross_val_qini,
    fit_response_model,
    save_metrics,
    select_best_cv,
    train_all,
)
from uplift.plots import (
    plot_decile_calibration,
    plot_qini_curves,
    plot_segments,
    plot_uplift_distribution,
)
from uplift.segments import classify, summarize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="hillstrom")
    p.add_argument("--outcome", default=None, help="override the dataset's primary outcome")
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--n-boot", type=int, default=200, help="bootstrap draws for the Qini CI")
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--cv-repeats", type=int, default=2)
    p.add_argument("--no-cv", action="store_true", help="select on the holdout instead (noisy)")
    p.add_argument("--no-causal-forest", action="store_true")
    p.add_argument("--mlflow-experiment", default="uplift", help="empty string disables tracking")
    args = p.parse_args()

    spec = get_spec(args.dataset)
    data = prepare(spec, outcome=args.outcome, sample_frac=args.sample_frac)
    outcome = args.outcome or spec.outcome_col
    print(f"\n{spec.name}: {data.n_train:,} train / {data.n_test:,} test, outcome={outcome}")
    print(f"treated share: train {data.t_train.mean():.1%}, test {data.t_test.mean():.1%}")

    # ---- Model selection: repeated CV on the training set only -----------
    cv_summary = None
    if not args.no_cv:
        print(f"\nCross-validating ({args.cv_splits}-fold x {args.cv_repeats}) for model selection...")
        cv_summary = cross_val_qini(
            data,
            include_causal_forest=not args.no_causal_forest,
            n_splits=args.cv_splits,
            n_repeats=args.cv_repeats,
        )
        print(f"\n{'='*96}\nCROSS-VALIDATED QINI (training folds — this is what selects the model)\n{'='*96}")
        print(
            cv_summary[["model", "qini_mean", "qini_se", "qini_std", "qini_min", "qini_max", "auuc_mean"]]
            .to_string(index=False, float_format=lambda v: f"{v:.4f}")
        )

    results = train_all(
        data,
        include_causal_forest=not args.no_causal_forest,
        n_boot=args.n_boot,
        mlflow_experiment=args.mlflow_experiment or None,
    )

    print(f"\n{'='*96}\nMODEL COMPARISON (holdout, outcome={outcome})\n{'='*96}")
    header = f"{'model':<17}{'Qini':>9}{'95% CI':>20}{'AUUC':>8}{'incr@20%':>10}{'neg share':>11}{'fit s':>8}"
    print(header + "\n" + "-" * len(header))
    for name, res in sorted(results.items(), key=lambda kv: -kv[1]["metrics"]["qini_coefficient"]):
        m = res["metrics"]
        ci = f"[{m['qini_ci_low']:+.4f}, {m['qini_ci_high']:+.4f}]"
        print(
            f"{name:<17}{m['qini_coefficient']:>9.4f}{ci:>20}{m['auuc']:>8.3f}"
            f"{m['uplift_at_20pct']:>10.1f}{m['pred_negative_share']:>11.1%}{m['fit_seconds']:>8.1f}"
        )

    if cv_summary is not None:
        best, separated = select_best_cv(cv_summary)
        runner_up = str(cv_summary.iloc[1]["model"])
        bm, rm_ = cv_summary.iloc[0], cv_summary.iloc[1]
        print(f"\nSelected on CV: {best}  (CV Qini {bm['qini_mean']:.4f} +/- {bm['qini_se']:.4f} SE)")
        print(
            f"  vs runner-up ({runner_up}, {rm_['qini_mean']:.4f}): "
            + (
                "separated by more than the combined standard error."
                if separated
                else "within combined standard error — a tie on this data. Breaking it on "
                "interpretability and serving cost, not the decimal place."
            )
        )
        print(f"  Holdout Qini for {best}: {results[best]['metrics']['qini_coefficient']:.4f} "
              f"(reported, not used for selection)")
    else:
        from uplift.pipeline import select_best

        best = select_best(results)
        runner_up = sorted(results, key=lambda k: -results[k]["metrics"]["qini_coefficient"])[1]
        b, r = results[best]["metrics"], results[runner_up]["metrics"]
        separated = b["qini_ci_low"] > r["qini_coefficient"]
        print(f"\nSelected on the holdout (--no-cv): {best}")
        print(
            f"  vs runner-up ({runner_up}): {b['qini_coefficient']:.4f} vs {r['qini_coefficient']:.4f} — "
            + ("separated" if separated else "NOT separated; the bootstrap CIs overlap")
        )

    # Reference points: the naive response model, and a random ranking. Both are
    # scored with exactly the same machinery as the uplift models.
    response_model = fit_response_model(data)
    response_scores = response_model.predict_proba(data.X_test)[:, 1]
    from uplift.evaluate import evaluate as _evaluate

    results["response_model"] = _evaluate(
        data.y_test, data.t_test, response_scores, name="response_model", n_boot=args.n_boot
    )
    rm = results["response_model"]["metrics"]
    print(
        f"\nNaive response model, ranked the same way: Qini {rm['qini_coefficient']:.4f} "
        f"[{rm['qini_ci_low']:+.4f}, {rm['qini_ci_high']:+.4f}]"
    )
    print("  It predicts who converts, so it cannot help but rank sure things first.")

    bundle = build_bundle(data, results, best, spec)
    path = bundle.save()
    print(f"\nSaved {path}")

    # ---- Segmentation on the holdout -------------------------------------
    baseline = bundle.baseline_model
    tau_test = results[best]["uplift_test"]
    p_control = baseline.predict_proba(data.X_test)[:, 1]
    segs = classify(
        tau_test, p_control, bundle.uplift_cut, bundle.baseline_cut, bundle.sleeping_dog_threshold
    )
    seg_summary = summarize(segs, data.y_test, data.t_test, data.value_test)
    print(f"\n{'='*96}\nSEGMENTS ON THE HOLDOUT (observed uplift, not predicted)\n{'='*96}")
    cols = ["segment", "n", "share", "rate_treated", "rate_control", "observed_uplift", "ci_low", "ci_high", "significant"]
    print(seg_summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    sd = seg_summary[seg_summary["segment"] == "sleeping_dog"]
    if not len(sd):
        print(f"\n  No sleeping dogs: the model predicts a negative effect for 0 of {data.n_test:,}")
        print("  holdout customers. Reported as found — on this campaign the treatment helps")
        print("  everywhere, and the interesting heterogeneity is in magnitude, not sign.")
    else:
        row = sd.iloc[0]
        if row["ci_high"] < 0:
            print(f"\n  Sleeping dogs confirmed: {row['n']:,} customers, observed uplift "
                  f"{row['observed_uplift']:+.4f} with the CI entirely below zero.")
        else:
            print(f"\n  {row['n']:,} customers flagged as sleeping dogs, but the observed CI "
                  f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] includes zero —")
            print("  this holdout cannot confirm the effect is actually negative for them.")

    ranked = seg_summary.sort_values("observed_uplift", ascending=False)
    print(f"\n  Ordering check: segments ranked by observed uplift -> {', '.join(ranked['segment'])}")
    st = seg_summary[seg_summary["segment"] == "sure_thing"]
    if len(st):
        row = st.iloc[0]
        print(
            f"  Sure things behave as labelled: highest do-nothing rate ({row['rate_control']:.1%}) "
            f"and an incremental effect of {row['observed_uplift']:+.4f} "
            + ("that is not distinguishable from zero." if not row["significant"] else "that remains significant.")
        )
        print("  Contacting them is close to pure cost — this is the budget the policy frees up.")

    # ---- Figures ----------------------------------------------------------
    # Figures are namespaced by dataset so a Criteo run cannot silently
    # overwrite the Hillstrom figures the README links to.
    pre = spec.name
    curves = {n: results[n]["curve"] for n in results if n != "response_model"}
    plot_qini_curves(curves, f"Qini curves — {spec.name}, outcome: {outcome}",
                     name=f"{pre}_qini_curves.png")
    plot_decile_calibration(
        decile_table(data.y_test, data.t_test, tau_test),
        f"Predicted vs. observed uplift by decile — {best}", name=f"{pre}_deciles.png",
    )
    plot_segments(seg_summary, f"Four-box segmentation — {spec.name}", name=f"{pre}_segments.png")
    plot_uplift_distribution(tau_test, f"Predicted uplift distribution — {best}",
                             name=f"{pre}_uplift_distribution.png")

    save_metrics(
        results,
        REPORTS / f"{spec.name}_metrics.json",
        extra={
            "dataset": spec.name,
            "outcome": outcome,
            "best_model": best,
            "runner_up": runner_up,
            "best_separated_from_runner_up": bool(separated),
            "selection": "repeated_cv" if cv_summary is not None else "holdout",
            "cv_summary": (
                json.loads(cv_summary.to_json(orient="records")) if cv_summary is not None else None
            ),
            "n_train": data.n_train,
            "n_test": data.n_test,
            "treated_share": float(data.t_train.mean()),
            "segments": json.loads(seg_summary.to_json(orient="records")),
        },
    )
    print(f"\nWrote {REPORTS / f'{spec.name}_metrics.json'} and figures to {FIGURES}")


if __name__ == "__main__":
    main()
