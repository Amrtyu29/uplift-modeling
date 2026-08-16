"""Days 7-8: turn uplift scores into a targeting policy and price it.

Compares four policies on the same randomized holdout:
  uplift          — contact the top k% by predicted incremental effect
  response_model  — contact the top k% by predicted conversion probability
  random          — contact a random k%, averaged over draws
  treat_all       — contact everyone (what most campaigns actually do)

    python scripts/03_simulate.py --dataset hillstrom
    python scripts/03_simulate.py --dataset hillstrom --cost-per-contact 0.50
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from uplift.config import REPORTS, Economics, get_spec
from uplift.data import prepare
from uplift.pipeline import UpliftBundle, fit_response_model, load_default_bundle
from uplift.plots import plot_break_even, plot_money_chart, plot_naive_vs_uplift
from uplift.simulate import (
    bootstrap_profit_curve,
    break_even_contact_cost,
    compare_policies,
    headline,
    optimal_depth,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="hillstrom")
    p.add_argument("--model-path", default=None)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--value-per-conversion", type=float, default=None)
    p.add_argument("--cost-per-contact", type=float, default=None)
    p.add_argument("--n-boot", type=int, default=200, help="bootstrap draws for the profit CI")
    p.add_argument("--profit-basis", choices=["observed", "modeled"], default=None,
                   help="observed = dataset's own revenue column; modeled = value_per_conversion")
    args = p.parse_args()

    spec = get_spec(args.dataset)
    bundle = UpliftBundle.load(args.model_path) if args.model_path else load_default_bundle(args.dataset)
    data = prepare(spec, sample_frac=args.sample_frac)

    econ = Economics(
        value_per_conversion=args.value_per_conversion or spec.economics.value_per_conversion,
        cost_per_contact=args.cost_per_contact or spec.economics.cost_per_contact,
        currency=spec.economics.currency,
    )
    # Prefer the dataset's own revenue column when it has one: it needs no
    # assumption about what a conversion is worth.
    basis = args.profit_basis or ("observed" if data.value_test is not None else "modeled")
    profit_col = "profit_observed" if basis == "observed" else "profit_modeled"

    print(f"\nPolicy simulation — {spec.name}, model={bundle.model_name}, n_holdout={data.n_test:,}")
    print(f"Economics: {econ.currency}{econ.value_per_conversion:g}/conversion, "
          f"{econ.currency}{econ.cost_per_contact:g}/contact")
    print(f"Profit basis: {basis}" + (
        f" (uses the dataset's `{spec.revenue_col}` column — no assumed conversion value)"
        if basis == "observed" else " (prices conversions at the assumed value)"))

    scores = bundle.predict(data.X_test)
    response_model = fit_response_model(data)
    policies = {
        "uplift": scores["uplift"].to_numpy(),
        "response_model": response_model.predict_proba(data.X_test)[:, 1],
    }

    comparison = compare_policies(
        policies, data.y_test, data.t_test, econ, value=data.value_test
    )

    print(f"\n{'='*104}\nPROFIT BY TARGETING DEPTH ({econ.currency}, holdout of {data.n_test:,})\n{'='*104}")
    pivot = comparison.pivot_table(index="depth", columns="policy", values=profit_col)
    order = [c for c in ["uplift", "response_model", "random", "treat_all"] if c in pivot.columns]
    print(pivot[order].to_string(float_format=lambda v: f"{v:,.0f}"))

    h = headline(comparison, profit_col)
    print(f"\n{'='*104}\nHEADLINE\n{'='*104}")
    print(f"  Optimal depth              : top {h['best_depth']*100:.0f}%  ({h['n_targeted']:,} customers)")
    print(f"  Incremental conversions    : {h['incremental_conversions']:,.0f}")
    print(f"  Incremental revenue        : {econ.currency}{h['incremental_value']:,.0f}")
    print(f"  Contact cost               : {econ.currency}{h['cost']:,.0f}")
    print(f"  Profit at optimum          : {econ.currency}{h['profit']:,.0f}")
    print(f"  Profit if treating everyone: {econ.currency}{h['profit_treat_all']:,.0f}")
    print(f"  Profit, random at same depth: {econ.currency}{h['profit_random_same_depth']:,.0f}")
    print(f"\n  vs. treat-everyone         : {h['gain_vs_treat_all_pct']:+.1f}% profit, "
          f"{h['contacts_saved_vs_treat_all']*100:.0f}% fewer contacts")
    print(f"  vs. random at same depth   : {h['gain_vs_random_pct']:+.1f}% profit")

    resp_curve = comparison[comparison["policy"] == "response_model"]
    resp_best = optimal_depth(resp_curve, profit_col)
    print(f"  vs. response model at its own optimum (top {resp_best['best_depth']*100:.0f}%): "
          f"{econ.currency}{resp_best['profit']:,.0f} -> uplift policy is "
          f"{(h['profit']-resp_best['profit'])/abs(resp_best['profit'])*100:+.1f}%")

    # Incremental conversions is the model-quality view; profit is the decision
    # view. They peak at different depths whenever contact is not free, which is
    # the entire reason the cost assumption has to be stated.
    inc = comparison[comparison["policy"] == "uplift"]
    peak_inc = inc.loc[inc["incremental_conversions"].idxmax()]
    print(f"\n  Note: incremental conversions keep rising to top {peak_inc['depth']*100:.0f}%, "
          f"but profit peaks at top {h['best_depth']*100:.0f}%.")
    print("  Where those two disagree, the contact cost is deciding the policy.")

    # ---- How much of this is real? ---------------------------------------
    band = bootstrap_profit_curve(
        policies["uplift"], data.y_test, data.t_test, econ, data.value_test, n_boot=args.n_boot
    )
    at_best = band[np.isclose(band["depth"], h["best_depth"])]
    if len(at_best):
        lo, hi = at_best[f"{profit_col}_lo"].iloc[0], at_best[f"{profit_col}_hi"].iloc[0]
        print(f"\n  Profit at the optimum, 95% CI: [{econ.currency}{lo:,.0f}, {econ.currency}{hi:,.0f}]")
        if lo < h["profit_treat_all"] < hi:
            print("  That interval contains the treat-everyone profit — on this holdout the")
            print("  advantage is real in point estimate but not statistically separated.")

    # ---- Break-even: when is the model worth running at all? -------------
    sweep = break_even_contact_cost(
        policies["uplift"], data.y_test, data.t_test, econ, data.value_test,
        basis=basis, n_boot=args.n_boot,
    )
    wins = sweep[sweep["uplift_wins"]]
    print(f"\n{'='*104}\nBREAK-EVEN ON CONTACT COST\n{'='*104}")
    print("  Selective targeting only pays when contacting someone costs enough to be")
    print("  worth withholding. Sweeping the cost instead of assuming a flattering one:\n")
    # Must follow the profit basis: datasets without a revenue column (Criteo)
    # have no incremental_value at all, and dividing it gives NaN.
    gross = (
        h["incremental_value"] if basis == "observed"
        else h["incremental_conversions"] * econ.value_per_conversion
    )
    rev_per_contact = gross / h["n_targeted"]
    print(f"  Incremental revenue per contact at the optimum: {econ.currency}{rev_per_contact:.3f}")
    if len(wins):
        be = wins["cost_per_contact"].min()
        row = wins.iloc[0]
        print(f"  Uplift targeting beats treat-everyone from {econ.currency}{be:.2f}/contact upward")
        print(f"    (at that cost: target the top {row['best_depth']*100:.0f}%, "
              f"{econ.currency}{row['gain_vs_treat_all']:,.0f} more profit, "
              f"95% CI [{econ.currency}{row['gain_lo']:,.0f}, {econ.currency}{row['gain_hi']:,.0f}])")
        print("    'Beats' here means the bootstrap lower bound clears zero, not just the point estimate.")
        expensive = sweep[sweep["cost_per_contact"] >= be * 3]
        if len(expensive):
            r = expensive.iloc[0]
            # The percentage is only meaningful while the blanket campaign is
            # still comfortably profitable. As its profit approaches zero the
            # ratio explodes — it swings by hundreds of percent across a couple
            # of cents of assumed cost — so past that point report dollars only.
            stable_pct = r["profit_treat_all"] > 0.2 * r["profit_uplift"]
            tail = (
                f"{r['gain_pct']:+.0f}% vs treat-everyone ({econ.currency}{r['gain_vs_treat_all']:,.0f})"
                if stable_pct
                else f"{econ.currency}{r['gain_vs_treat_all']:,.0f} more profit "
                     f"(treat-everyone earns {econ.currency}{r['profit_treat_all']:,.0f} here, so the "
                     f"percentage is unstable and omitted)"
            )
            print(f"  At {econ.currency}{r['cost_per_contact']:.2f}/contact (direct-mail territory): "
                  f"top {r['best_depth']*100:.0f}%, {tail}")
        loss_making = sweep[sweep["profit_treat_all"] < 0]
        if len(loss_making):
            r = loss_making.iloc[0]
            print(f"  Past {econ.currency}{r['cost_per_contact']:.2f}/contact a blanket campaign loses "
                  f"money ({econ.currency}{r['profit_treat_all']:,.0f}) while targeting the top "
                  f"{r['best_depth']*100:.0f}% still earns {econ.currency}{r['profit_uplift']:,.0f}.")
    else:
        print("  Treat-everyone is not beaten at any cost swept, once sampling noise is accounted for.")
    print(f"\n  At the assumed {econ.currency}{econ.cost_per_contact:g}/contact, the optimum is "
          f"top {h['best_depth']*100:.0f}%.")
    if econ.cost_per_contact < (wins["cost_per_contact"].min() if len(wins) else np.inf):
        print("  At this cost the channel is cheap enough that broad targeting is close to")
        print("  optimal, and saying so is the correct finding rather than a failed model.")
        print("  The ranking's value is knowing who to drop first, which is what pays off on")
        print("  any channel with a real per-contact cost or a customer-fatigue budget.")

    plot_money_chart(
        comparison, econ, profit_col,
        title=f"Incremental profit vs. % targeted — {spec.name} ({basis} revenue)",
        band=band, name=f"{spec.name}_money_chart.png",
    )
    plot_naive_vs_uplift(comparison, econ, name=f"{spec.name}_naive_vs_uplift.png")
    plot_break_even(sweep, econ, current_cost=econ.cost_per_contact,
                    name=f"{spec.name}_break_even.png")

    out = REPORTS / f"{spec.name}_simulation.json"
    out.write_text(
        json.dumps(
            {
                "dataset": spec.name,
                "model": bundle.model_name,
                "n_holdout": data.n_test,
                "economics": {
                    "value_per_conversion": econ.value_per_conversion,
                    "cost_per_contact": econ.cost_per_contact,
                    "currency": econ.currency,
                },
                "profit_basis": basis,
                "headline": h,
                "response_model_optimum": resp_best,
                "profit_ci_at_optimum": (
                    {"low": float(at_best[f"{profit_col}_lo"].iloc[0]),
                     "high": float(at_best[f"{profit_col}_hi"].iloc[0])} if len(at_best) else None
                ),
                "break_even_cost_per_contact": (
                    float(wins["cost_per_contact"].min()) if len(wins) else None
                ),
                "incremental_revenue_per_contact": float(rev_per_contact),
                "cost_sweep": json.loads(sweep.to_json(orient="records")),
                "curves": json.loads(comparison.to_json(orient="records")),
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
