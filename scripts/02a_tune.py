"""Grid search over base-learner regularization, scored by cross-validated Qini.

This exists to justify one non-obvious choice: the base learners are regularized
far harder than they would be for ordinary classification. Uplift is a
difference of two small probabilities, so variance in either arm lands straight
on the estimate instead of averaging out. The grid below makes that concrete —
deeper trees fit the outcome better and rank the treatment effect worse.

    python scripts/02a_tune.py --dataset hillstrom

Writes reports/<dataset>_tuning.json. The winning row is what lives in
`uplift.learners.DEFAULT_PARAMS`.
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from uplift import learners as L
from uplift.config import REPORTS, get_spec
from uplift.data import prepare
from uplift.evaluate import qini_coefficient, qini_curve

GRID = [
    {"num_leaves": 31, "min_child_samples": 100, "n_estimators": 300},
    {"num_leaves": 15, "min_child_samples": 200, "n_estimators": 200},
    {"num_leaves": 7, "min_child_samples": 400, "n_estimators": 150},
    {"num_leaves": 7, "min_child_samples": 800, "n_estimators": 100},
    {"num_leaves": 4, "min_child_samples": 1000, "n_estimators": 60},
    {"num_leaves": 4, "min_child_samples": 2000, "n_estimators": 40},
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="hillstrom")
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--n-splits", type=int, default=4)
    p.add_argument("--n-repeats", type=int, default=2)
    args = p.parse_args()

    spec = get_spec(args.dataset)
    data = prepare(spec, sample_frac=args.sample_frac)
    X, t, y = data.X_train, data.t_train, data.y_train
    strata = t * 2 + y
    cv = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=0)
    folds = list(cv.split(X, strata))

    original = dict(L.DEFAULT_PARAMS)
    rows = []
    try:
        for params in GRID:
            label = f"lv{params['num_leaves']}_mcs{params['min_child_samples']}_ne{params['n_estimators']}"
            # The learner classes read DEFAULT_PARAMS at construction time, so
            # patching it here is what makes every learner share the grid point.
            L.DEFAULT_PARAMS.update(params)
            for fold, (tr, va) in enumerate(folds):
                for name, learner in L.build_learners(include_causal_forest=False).items():
                    learner.fit(X.iloc[tr], t[tr], y[tr])
                    tau = learner.predict_uplift(X.iloc[va])
                    rows.append(
                        {
                            "grid": label,
                            **params,
                            "model": name,
                            "fold": fold,
                            "qini": qini_coefficient(qini_curve(y[va], t[va], tau)),
                        }
                    )
            done = pd.DataFrame(rows).query("grid == @label").groupby("model")["qini"].mean()
            print(f"{label:<28} " + "  ".join(f"{k}={v:.4f}" for k, v in done.items()))
    finally:
        L.DEFAULT_PARAMS.clear()
        L.DEFAULT_PARAMS.update(original)

    df = pd.DataFrame(rows)
    mean = df.pivot_table(index="grid", columns="model", values="qini", aggfunc="mean")
    se = df.pivot_table(index="grid", columns="model", values="qini", aggfunc="sem")

    print(f"\n{'='*80}\nMEAN CV QINI BY GRID POINT\n{'='*80}")
    print(mean.round(4).to_string())
    print(f"\nStandard error (differences smaller than ~{se.to_numpy().mean()*2:.3f} are noise)")
    print(se.round(4).to_string())

    best_overall = mean.mean(axis=1).idxmax()
    best_single = mean.stack().idxmax()
    print(f"\nBest grid point averaged over learners : {best_overall}")
    print(f"Best single (grid, learner) combination: {best_single[0]} / {best_single[1]} "
          f"= {mean.stack().max():.4f}")

    out = REPORTS / f"{spec.name}_tuning.json"
    out.write_text(
        json.dumps(
            {
                "dataset": spec.name,
                "grid": GRID,
                "mean_qini": json.loads(mean.to_json()),
                "sem_qini": json.loads(se.to_json()),
                "best_grid_overall": best_overall,
                "best_grid_single": list(best_single),
                "folds": args.n_splits * args.n_repeats,
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
