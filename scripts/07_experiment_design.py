"""Stretch goal: design the live experiment that would validate this model.

The simulation says what the policy would have earned on historical randomized
data. Running it for real is a separate decision, and it needs a test design
with an honest sample size attached. This script produces that design from the
model's own measured numbers rather than from assumed ones.

    python scripts/07_experiment_design.py
    python scripts/07_experiment_design.py --depth 0.3 --daily-traffic 5000
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np

from uplift.config import REPORTS, get_spec
from uplift.data import average_treatment_effect, prepare
from uplift.experiment import (
    design_decay_monitor,
    design_policy_test,
    design_profit_test,
    duration_table,
    mde_for_sample_size,
    power_curve,
    sample_size_two_proportions,
)
from uplift.pipeline import load_default_bundle
from uplift.plots import plot_power_analysis


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="hillstrom")
    p.add_argument("--depth", type=float, default=None,
                   help="targeting depth to test; defaults to the simulated optimum")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.8)
    p.add_argument("--daily-traffic", type=int, default=2000)
    p.add_argument("--sample-frac", type=float, default=None)
    args = p.parse_args()

    spec = get_spec(args.dataset)
    bundle = load_default_bundle(args.dataset)
    data = prepare(spec, sample_frac=args.sample_frac)
    uplift = bundle.predict(data.X_test)["uplift"].to_numpy()

    ate = average_treatment_effect(data.y_test, data.t_test)
    print(f"\nExperiment design — {spec.name}, model {bundle.model_name}")
    print(f"Measured on the holdout: control rate {ate['rate_control']:.4f}, "
          f"treatment effect {ate['ate']:+.4f}")
    print(f"alpha={args.alpha}, power={args.power:.0%}\n")

    # ---- Test 1: does the treatment work at all? -------------------------
    print(f"{'='*100}\nTEST 1 — DOES THE TREATMENT WORK AT ALL?  (treated vs. control)\n{'='*100}")
    t1 = sample_size_two_proportions(
        ate["rate_control"], ate["rate_control"] + ate["ate"], alpha=args.alpha, power=args.power
    )
    print(f"  Detecting the measured effect of {ate['ate']:+.4f} "
          f"({ate['ate']/ate['rate_control']:+.0%} relative)")
    print(f"  Sample size: {t1['n_total']:,} total ({t1['n_control']:,} per arm)")
    print("  This is the easy one, and it is already answered by the source experiment.")

    # ---- Test 2: does targeting beat the blanket campaign? ---------------
    depth = args.depth
    if depth is None:
        sim_path = REPORTS / f"{spec.name}_simulation.json"
        depth = (
            json.loads(sim_path.read_text())["headline"]["best_depth"]
            if sim_path.exists() else 0.3
        )
    print(f"\n{'='*100}\nTEST 2 — DOES UPLIFT TARGETING BEAT THE BLANKET CAMPAIGN?\n{'='*100}")
    print(f"  Arm A: contact everyone (the incumbent)")
    print(f"  Arm B: contact the top {depth:.0%} by predicted uplift\n")

    t2 = design_policy_test(data.y_test, data.t_test, uplift, depth, args.alpha, args.power)
    print("  (a) Powered on RESPONSE RATE — the intuitive choice, and the wrong one:\n")
    print(f"      Response rate under A     : {t2['rate_policy_a_treat_all']:.4f}")
    print(f"      Response rate under B     : {t2['rate_policy_b_targeted']:.4f}")
    print(f"      Difference to detect      : {t2['expected_difference']:+.5f}")
    print(f"      Sample size               : {t2['n_total']:,} total "
          f"({t2['n_total']/max(t1['n_total'],1):,.0f}x Test 1)")
    print(f"\n      Note the sign. Every subgroup in this data has a positive effect, so")
    print(f"      withholding contact from {t2['share_dropped']:.0%} of the file can only LOWER the overall")
    print("      response rate — that is the population, not a bad model. A test powered this")
    print("      way is designed to conclude that targeting hurts: it would be right about the")
    print("      metric it measured and useless for the decision being made, because the gains")
    print("      from targeting are entirely in the costs avoided, which this metric cannot see.")
    print("      (Where real sleeping dogs exist the sign flips and the same test flatters the")
    print("      policy instead — wrong in the opposite direction, for the same reason.)")

    # Same revenue basis the simulation used, so the design sizes a test for the
    # conclusion the simulation actually reached.
    t2p = design_profit_test(
        data.y_test, data.t_test, uplift, depth,
        spec.economics.value_per_conversion, spec.economics.cost_per_contact,
        args.alpha, args.power, value=data.value_test,
    )
    cur = spec.economics.currency
    print(f"\n  (b) Powered on PROFIT PER CUSTOMER — the outcome the decision actually turns on:")
    print(f"      (revenue basis: {t2p.get('basis', 'n/a')})\n")
    if t2p.get("status"):
        print(f"      {t2p['status']}")
    else:
        print(f"      Profit/customer under A   : {cur}{t2p['profit_per_customer_a']:.4f}")
        print(f"      Profit/customer under B   : {cur}{t2p['profit_per_customer_b']:.4f}")
        print(f"      Difference to detect      : {cur}{t2p['difference']:+.4f} per customer "
              f"({'B better' if t2p['b_is_better'] else 'A better'})")
        print(f"      Per-customer sd           : {cur}{t2p['sd_a']:.2f} (A), {cur}{t2p['sd_b']:.2f} (B)")
        print(f"      Sample size               : {t2p['n_total']:,} total "
              f"({t2p['n_per_arm']:,} per arm)")
        print("\n      Profit is a noisier outcome than a rate, so this is not the cheaper test —")
        print("      it is the one whose answer means something. Sizing on the wrong metric is")
        print("      how a project gets a statistically clean result that settles nothing.")

    print("\n  Both designs are large for the same underlying reason: the two arms contact the")
    print(f"  same top slice, so those customers contribute nothing. All the signal comes from")
    print(f"  the {t2['share_dropped']:.0%} that policy B withholds. That is the number deciding whether")
    print("  this is testable at all, and it is much cheaper to learn now than after six")
    print("  inconclusive weeks.")

    design_n = t2p["n_total"] if not t2p.get("status") else t2["n_total"]
    dur = duration_table(design_n, [args.daily_traffic, args.daily_traffic * 5,
                                         args.daily_traffic * 25, args.daily_traffic * 100])
    print(f"\n  {'daily traffic':>15}{'days':>10}{'weeks':>9}{'in a quarter?':>16}")
    for _, r in dur.iterrows():
        print(f"  {int(r['daily_traffic']):>15,}{r['days']:>10.0f}{r['weeks']:>9.1f}"
              f"{('yes' if r['feasible_in_a_quarter'] else 'no'):>16}")

    # A cheaper alternative worth putting on the table.
    print(f"\n  If that is out of reach, test a deeper cut (profit basis, same as (b)):")
    print(f"  {'depth':>8}{'profit diff/customer':>24}{'better':>9}{'n total':>14}")
    alt_rows = []
    for d in [0.2, 0.3, 0.5, 0.7, 0.9]:
        alt = design_profit_test(
            data.y_test, data.t_test, uplift, d,
            spec.economics.value_per_conversion, spec.economics.cost_per_contact,
            args.alpha, args.power, value=data.value_test,
        )
        if alt.get("status"):
            continue
        alt_rows.append(alt)
        print(f"  {d:>8.0%}{cur + format(alt['difference'], '+.4f'):>24}"
              f"{('B' if alt['b_is_better'] else 'A'):>9}{alt['n_total']:>14,}")
    print("\n  Withholding more customers makes the difference larger and the test cheaper —")
    print("  at the cost of testing a policy further from the one you actually want to run.")

    # ---- Is this testable at any realistic contact cost? -----------------
    # The break-even sweep in 03_simulate showed the policy's advantage grows
    # with contact cost. If that is true, the test should get cheaper too — and
    # that is the design argument for when to run it.
    print(f"\n{'='*100}\nAT WHAT CONTACT COST DOES THIS BECOME TESTABLE?\n{'='*100}")
    print(f"  {'cost/contact':>14}{'best depth':>12}{'profit diff':>14}{'better':>9}{'n total':>16}")
    cost_rows = []
    for c in [spec.economics.cost_per_contact, 0.25, 0.50, 0.74, 1.00, 2.00]:
        best = None
        for d in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]:
            cand = design_profit_test(
                data.y_test, data.t_test, uplift, d,
                spec.economics.value_per_conversion, c, args.alpha, args.power,
                value=data.value_test,
            )
            if cand.get("status"):
                continue
            if best is None or cand["difference"] > best["difference"]:
                best = cand
        if best is None:
            continue
        best["cost_per_contact"] = c
        cost_rows.append(best)
        print(f"  {cur + format(c, '.2f'):>14}{best['depth']:>11.0%}"
              f"{cur + format(best['difference'], '+.4f'):>14}"
              f"{('B' if best['b_is_better'] else 'A'):>9}{best['n_total']:>16,}")

    testable = [r for r in cost_rows if r["b_is_better"] and r["n_total"] <= 200_000]
    print(f"\n  VERDICT")
    if testable:
        r = min(testable, key=lambda x: x["n_total"])
        print(f"    At {cur}{r['cost_per_contact']:.2f}/contact the advantage is "
              f"{cur}{r['difference']:.4f} per customer and needs {r['n_total']:,} customers —")
        print(f"    a test you can actually book. Run the experiment on an expensive channel,")
        print(f"    targeting the top {r['depth']:.0%}, not on cheap email.")
    else:
        print("    None of these costs make the comparison testable at a realistic sample size.")
    print(f"\n    On cheap email the honest answer is that the model's edge is real in point")
    print(f"    estimate and too small to prove: {t2p['n_total']:,} customers to detect")
    print(f"    {cur}{t2p['difference']:+.4f} per head. That is the same conclusion the profit")
    print("    confidence interval reached in 03_simulate, arrived at from the other direction,")
    print("    and it is a design finding rather than a modelling failure — the experiment to")
    print("    justify this project has to be run where the money is, not where the data is.")

    # ---- What can the traffic you have actually detect? ------------------
    print(f"\n{'='*100}\nWHAT CAN A REALISTIC TEST ACTUALLY DETECT?\n{'='*100}")
    for weeks in (2, 4, 8):
        n = args.daily_traffic * 7 * weeks
        m = mde_for_sample_size(n, ate["rate_control"], args.alpha, args.power)
        verdict = "enough" if m["mde_absolute"] <= abs(t2["expected_difference"]) else "NOT enough"
        print(f"  {weeks:>2} weeks at {args.daily_traffic:,}/day = {n:,} users -> "
              f"smallest detectable difference {m['mde_absolute']:.5f} "
              f"({m['mde_relative']:+.1%} relative) — {verdict} for Test 2")

    # ---- Test 3: the always-on decay monitor -----------------------------
    print(f"\n{'='*100}\nTEST 3 — THE ALWAYS-ON HOLDOUT THAT CATCHES EFFECT DECAY\n{'='*100}")
    decay_rows = []
    for decay in (0.2, 0.3, 0.5):
        d3 = design_decay_monitor(ate["ate"], ate["rate_control"], decay, args.alpha, args.power)
        decay_rows.append(d3)
        print(f"  Detect a {decay:.0%} drop in effect: {d3['n_total_holdout']:,} customers in the holdout")
    print("\n  This is the piece that is usually skipped, and the one that keeps every")
    print("  future retraining valid. Without it, the model's own targeting decisions")
    print("  confound next quarter's training data and the confounding is invisible.")

    # ---- Figure ----------------------------------------------------------
    curve = power_curve(
        t2["rate_policy_a_treat_all"], t2["expected_difference"],
        np.unique(np.geomspace(1_000, max(t2["n_total"] * 3, 10_000), 60).astype(int)),
        alpha=args.alpha,
    )
    plot_power_analysis(curve, t2["n_total"], args.power, name=f"{spec.name}_power_analysis.png")

    out = REPORTS / f"{spec.name}_experiment_design.json"
    out.write_text(
        json.dumps(
            {
                "dataset": spec.name,
                "model": bundle.model_name,
                "alpha": args.alpha,
                "power": args.power,
                "measured": ate,
                "test_1_treatment_effect": t1,
                "test_2_policy_comparison_response_rate": t2,
                "test_2_policy_comparison_profit": t2p,
                "test_2_alternative_depths": alt_rows,
                "test_3_decay_monitor": decay_rows,
                "testability_by_cost": cost_rows,
                "duration": json.loads(dur.to_json(orient="records")),
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
