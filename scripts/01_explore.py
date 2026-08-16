"""Day 1-2: check the randomization, measure the average effect, set the baseline.

Everything downstream assumes the treatment was randomly assigned. This script
is where that assumption gets tested rather than asserted, and where the naive
"predict conversion" framing is shown to answer a different question than the
one the campaign is asking.

    python scripts/01_explore.py --dataset hillstrom
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from uplift.config import REPORTS, get_spec
from uplift.data import average_treatment_effect, load_raw, prepare, randomization_check


def _rank_reversals(spec, sample_frac=None, min_cell: int = 500) -> dict:
    """Find variables where response ranking and uplift ranking disagree.

    For each candidate variable, bin it (quartiles if continuous), then compare
    the group ordering by treated-arm response rate against the ordering by
    observed uplift. A reversal is a direct, model-free demonstration that
    "most likely to convert" and "most moved by the offer" are different
    questions — and it is measured, not assumed.
    """
    df = load_raw(spec, sample_frac=sample_frac)
    if spec.treated_values is not None:
        df = df[df[spec.treatment_col].isin(spec.treated_values + (spec.control_values or []))]
        df["_t"] = df[spec.treatment_col].isin(spec.treated_values).astype(int)
    else:
        df["_t"] = df[spec.treatment_col].astype(int)

    candidates = list(spec.categorical_features) + list(spec.numeric_features)
    findings, tables = [], {}
    for col in candidates:
        s = df[col]
        key = s if (s.dtype == object or s.nunique() <= 5) else pd.qcut(s, 4, duplicates="drop")
        tab = df.groupby([key, "_t"], observed=True)[spec.outcome_col].mean().unstack()
        if tab.shape[1] < 2:
            continue
        tab.columns = ["control_rate", "treated_rate"]
        tab["uplift"] = tab["treated_rate"] - tab["control_rate"]
        tab["n"] = df.groupby(key, observed=True).size()
        tab = tab[tab["n"] >= min_cell]
        if len(tab) < 2:
            continue
        best_response, best_uplift = tab["treated_rate"].idxmax(), tab["uplift"].idxmax()
        tables[col] = tab
        if best_response != best_uplift:
            findings.append(
                {
                    "variable": col,
                    "top_by_response": str(best_response),
                    "top_by_uplift": str(best_uplift),
                    "uplift_of_response_pick": float(tab.loc[best_response, "uplift"]),
                    "uplift_of_uplift_pick": float(tab.loc[best_uplift, "uplift"]),
                    "table": json.loads(tab.reset_index().astype({col: str}).to_json(orient="records")),
                }
            )

    if not findings:
        print("  No single-variable reversals found — the divergence, if any, is multivariate.")
    for f in sorted(findings, key=lambda d: d["uplift_of_uplift_pick"] - d["uplift_of_response_pick"], reverse=True):
        print(f"  [{f['variable']}]")
        print(tables[f["variable"]].to_string(float_format=lambda v: f"{v:.4f}"))
        gap = f["uplift_of_uplift_pick"] - f["uplift_of_response_pick"]
        print(
            f"    -> response model picks {f['top_by_response']!r} (uplift {f['uplift_of_response_pick']:+.4f});"
            f" uplift picks {f['top_by_uplift']!r} (uplift {f['uplift_of_uplift_pick']:+.4f});"
            f" gap {gap:+.4f}\n"
        )
    return {"n_reversals": len(findings), "reversals": findings}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="hillstrom")
    p.add_argument("--sample-frac", type=float, default=None)
    args = p.parse_args()

    spec = get_spec(args.dataset)
    # Same sampling for both loads, so the row counts printed below are
    # comparable rather than describing two different subsets.
    data = prepare(spec, sample_frac=args.sample_frac)
    raw = load_raw(spec, sample_frac=args.sample_frac)

    print(f"\n{'='*70}\nDATASET: {spec.name}\n{'='*70}")
    sampled = "" if not args.sample_frac else f" (sampled at {args.sample_frac:.0%})"
    print(f"rows loaded         : {len(raw):,}{sampled}")
    print(f"after arm filtering : {data.n_train + data.n_test:,}")
    print(f"train / test        : {data.n_train:,} / {data.n_test:,}")
    print(f"features            : {len(data.feature_names)}  {data.feature_names}")

    t_all = np.concatenate([data.t_train, data.t_test])
    print(f"\ntreated share       : {t_all.mean():.1%}  ({int(t_all.sum()):,} treated)")

    print(f"\n{'-'*70}\nRANDOMIZATION CHECK (standardized mean difference)\n{'-'*70}")
    balance = randomization_check(data.X_train, data.t_train)
    print(balance.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    n_imbalanced = int((~balance["balanced"]).sum())
    verdict = (
        "PASS — every |SMD| < 0.1, consistent with random assignment"
        if n_imbalanced == 0
        else f"CHECK — {n_imbalanced} feature(s) with |SMD| >= 0.1"
    )
    print(f"\n  {verdict}")
    print("  Note: with tens of thousands of rows, t-test p-values flag differences")
    print("  far too small to bias anything. SMD is the decision rule here.")

    print(f"\n{'-'*70}\nAVERAGE TREATMENT EFFECT — outcome: {spec.outcome_col}\n{'-'*70}")
    ate = average_treatment_effect(
        np.concatenate([data.y_train, data.y_test]), t_all
    )
    print(f"  treated rate      : {ate['rate_treated']:.4f}  (n={ate['n_treated']:,})")
    print(f"  control rate      : {ate['rate_control']:.4f}  (n={ate['n_control']:,})")
    print(f"  ATE               : {ate['ate']:+.4f}  95% CI [{ate['ate_ci_low']:+.4f}, {ate['ate_ci_high']:+.4f}]")
    print(f"  relative lift     : {ate['relative_lift']:+.1%}")
    print("\n  This is the campaign-level answer. The rest of the project is about")
    print("  whether that average hides groups it moves much more, or backwards.")

    alt_ates = {}
    for alt in spec.alt_outcome_cols:
        d_alt = prepare(spec, outcome=alt, sample_frac=args.sample_frac)
        # Must use this split's own treatment vector. `prepare` stratifies on
        # the outcome, so asking for a different outcome returns a *different*
        # train/test partition — pairing its y against the primary split's t
        # lines up mismatched rows and shrinks the estimate toward zero. It is
        # a silent failure: the output still looks like a plausible small ATE.
        a = average_treatment_effect(
            np.concatenate([d_alt.y_train, d_alt.y_test]),
            np.concatenate([d_alt.t_train, d_alt.t_test]),
        )
        alt_ates[alt] = a
        print(f"\n  [{alt}] treated {a['rate_treated']:.4f} vs control {a['rate_control']:.4f} "
              f"-> ATE {a['ate']:+.4f} (CI {a['ate_ci_low']:+.4f}, {a['ate_ci_high']:+.4f})")

    print(f"\n{'-'*70}\nWHY A RESPONSE MODEL ANSWERS THE WRONG QUESTION\n{'-'*70}")
    print("  Scanning single-variable splits for cases where ranking by response rate")
    print("  and ranking by uplift disagree. Where they do, a response model would")
    print("  spend the budget on the wrong group — no modelling required to show it.\n")
    naive = _rank_reversals(spec, args.sample_frac)

    out = REPORTS / f"{spec.name}_exploration.json"
    out.write_text(
        json.dumps(
            {
                "dataset": spec.name,
                "n_rows": int(data.n_train + data.n_test),
                "treated_share": float(t_all.mean()),
                "outcome": spec.outcome_col,
                "ate": ate,
                "alt_outcomes": alt_ates,
                "balance": json.loads(balance.to_json(orient="records")),
                "n_imbalanced_features": n_imbalanced,
                "naive_vs_uplift": naive,
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
