"""Day 11: drift monitoring on the treatment-effect distribution.

Simulates a sequence of incoming batches and checks each one against the
training-time uplift distribution. Three of the batches are deliberately
corrupted in ways that matter operationally:

  covariate shift  — the incoming audience changes
  effect decay     — the offer stops working; features look completely normal
  broken pipeline  — a feature arrives constant, e.g. an upstream join failure

Effect decay is the one a conventional monitor misses. Feature distributions are
untouched and the outcome model is fine; what changed is the response to
treatment. That is exactly what an uplift monitor is for.

    python scripts/04_monitor.py --dataset hillstrom
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from uplift.config import REPORTS, get_spec
from uplift.data import prepare
from uplift.monitoring import UpliftDriftMonitor, evidently_report, realized_effect
from uplift.pipeline import load_default_bundle
from uplift.plots import plot_drift


def pick_shift_feature(X: pd.DataFrame) -> str:
    """Choose a feature whose distribution genuinely moves the uplift score.

    Corrupting an ignored feature proves nothing — the monitor would correctly
    stay silent and the demo would look broken. The feature with the widest
    spread is the one most likely to matter to a tree model, and picking it by
    data rather than by name keeps this working on any dataset.
    """
    spread = X.std(numeric_only=True)
    return str(spread.idxmax())


def make_batches(
    X: pd.DataFrame, rng, n_batches: int = 8, size: int = 2000
) -> list[tuple[str, pd.DataFrame, str]]:
    """Build a batch sequence with known, labelled corruptions.

    Corruptions are chosen by data rather than by column name, so the same
    scenarios run on Hillstrom and on Criteo instead of silently no-op'ing on
    whichever dataset lacks the hardcoded column.
    """
    feature = pick_shift_feature(X)
    high = float(X[feature].quantile(0.9))
    batches = []
    for i in range(n_batches):
        sample = X.sample(size, replace=True, random_state=int(rng.integers(1e6))).reset_index(drop=True)
        label, kind = f"batch_{i+1:02d}", "clean"

        if i == 3:
            # Audience shift: the incoming population is drawn from one end of
            # the distribution, as after an acquisition push.
            sample[feature] = high
            label += " (covariate shift)"
            kind = "covariate_shift"
        elif i == 5:
            # Effect decay is not a data problem, so it cannot be simulated by
            # perturbing features. It is applied at the score level below.
            label += " (effect decay)"
            kind = "effect_decay"
        elif i == 6:
            # Upstream join failure: the column still arrives, but constant.
            sample[feature] = 0.0
            label += " (broken feature)"
            kind = "broken_feature"
        batches.append((label, sample, kind))
    return batches


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="hillstrom")
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--decay", type=float, default=0.45, help="fraction of the effect lost in the decay batch")
    p.add_argument("--sample-frac", type=float, default=None,
                   help="sample the holdout; Criteo is ~14M rows and needs this to be quick")
    args = p.parse_args()

    spec = get_spec(args.dataset)
    bundle = load_default_bundle(args.dataset)
    data = prepare(spec, sample_frac=args.sample_frac)
    rng = np.random.default_rng(7)

    monitor = UpliftDriftMonitor(bundle.reference_uplift)
    lo, hi = monitor.mean_band(args.batch_size)
    print(f"\nReference: {len(bundle.reference_uplift):,} training scores, "
          f"mean uplift {monitor.ref_mean:.4f} (sd {monitor.ref_std:.4f})")
    print(f"Tolerance band for a {args.batch_size:,}-row batch: [{lo:.4f}, {hi:.4f}]")
    print("Thresholds: PSI watch 0.10, alert 0.25\n")

    print(f"{'batch':<28}{'n':>7}{'mean tau':>11}{'PSI':>8}{'status':>9}  reason")
    print("-" * 100)

    planted = {}
    for label, batch, kind in make_batches(data.X_test, rng, args.n_batches, args.batch_size):
        tau = bundle.predict(batch)["uplift"].to_numpy()
        if kind == "effect_decay":
            # The offer's power fades: same customers, same features, smaller
            # true effect. A feature-drift monitor sees nothing here.
            tau = tau * (1 - args.decay)
        result = monitor.check(tau, batch=label)
        if kind != "clean":
            planted[kind] = result.status
        reason = result.reasons[0] if result.reasons else ""
        print(
            f"{label:<28}{result.n:>7,}{result.mean_uplift:>11.4f}{result.psi:>8.3f}"
            f"{result.status:>9}  {reason[:52]}"
        )

    history = monitor.history_frame()
    alerts = history[history["status"] != "OK"]
    print(f"\n{len(alerts)} of {len(history)} batches flagged.")
    for _, row in alerts.iterrows():
        print(f"\n  [{row['status']}] {row['batch']}")
        for r in row["reasons"]:
            print(f"      - {r}")

    caught = {k: v != "OK" for k, v in planted.items()}
    print(f"\nPlanted scenarios detected: {sum(caught.values())}/{len(caught)}")
    for kind, status in planted.items():
        print(f"  {kind:<17} {status}")
    print("\n  A feature-drift monitor would also have caught the covariate shift and the")
    print("  broken feature — both are visible in X. It would NOT have caught effect decay:")
    print("  features and outcome rates are unchanged there, and only the response to")
    print("  treatment moved. That is the failure mode that makes monitoring tau the point.")

    # Ground truth eventually arrives from the always-on randomized holdout.
    truth = realized_effect(data.y_test, data.t_test, bundle.predict(data.X_test)["uplift"].to_numpy())
    print(f"\n{'='*100}\nCLOSING THE LOOP ON THE RANDOMIZED HOLDOUT (top 30%)\n{'='*100}")
    print(f"  predicted uplift : {truth['predicted_uplift']:+.4f}")
    print(f"  observed uplift  : {truth['observed_uplift']:+.4f}  "
          f"(n_treated={truth['n_treated']:,}, n_control={truth['n_control']:,})")
    print(f"  calibration ratio: {truth['calibration_ratio']:.2f}  (1.0 = perfectly calibrated)")
    if truth["calibration_ratio"] < 1:
        print("  The model over-states the effect for its top slice — normal for a ranking")
        print("  model, and the reason the policy is chosen on the observed curve rather than")
        print("  on predicted uplift summed up.")

    log = monitor.write_log(REPORTS / f"{spec.name}_drift_log.jsonl")
    plot_drift(history, name=f"{spec.name}_drift_monitor.png")

    # Same shifted feature as the covariate-shift batch, so the optional
    # Evidently report describes the scenario this script actually planted.
    shift_feature = pick_shift_feature(data.X_test)
    ref_df = data.X_train.sample(min(5000, len(data.X_train)), random_state=0)
    cur_df = data.X_test.sample(min(5000, len(data.X_test)), random_state=1).assign(
        **{shift_feature: float(data.X_test[shift_feature].quantile(0.9))}
    )
    ev = evidently_report(ref_df, cur_df, REPORTS / f"{spec.name}_evidently.html")
    print(f"\nWrote {log}")
    print(f"Evidently feature-drift report: {ev if ev else 'skipped (evidently not installed)'}")

    (REPORTS / f"{spec.name}_monitoring.json").write_text(
        json.dumps(
            {
                "dataset": spec.name,
                "reference": {
                    "mean": monitor.ref_mean,
                    "sd": monitor.ref_std,
                    "batch_band": [lo, hi],
                    "n": len(bundle.reference_uplift),
                },
                "batches": json.loads(history.to_json(orient="records")),
                "n_alerts": int((history["status"] != "OK").sum()),
                "planted_scenarios": planted,
                "realized_effect": truth,
            },
            indent=2,
            default=float,
        )
    )
    print(f"Wrote {REPORTS / f'{spec.name}_monitoring.json'}")


if __name__ == "__main__":
    main()
