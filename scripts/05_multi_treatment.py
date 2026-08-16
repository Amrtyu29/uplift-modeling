"""Stretch goal: which of several offers works best for each customer.

Binary uplift answers "contact or not". Hillstrom actually ran two creatives —
a men's email and a women's email — against one shared control, so the sharper
question is available: *which* email, for whom.

One uplift model is fitted per arm against the shared control, and the policy
assigns each customer the arm with the highest predicted effect. The comparison
that matters is against the best single-arm blanket campaign, since that is what
a team without a model would run.

    python scripts/05_multi_treatment.py
"""

import argparse
import json

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from uplift.config import REPORTS, get_spec
from uplift.data import build_features, load_raw
from uplift.evaluate import qini_coefficient, qini_curve
from uplift.learners import SLearner
from uplift.plots import PALETTE, save_figure

CONTROL = "No E-Mail"
ARMS = ["Mens E-Mail", "Womens E-Mail"]


def observed_effect(y: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Uplift and its standard error for a treated/control comparison."""
    yt, yc = y[t == 1], y[t == 0]
    if len(yt) == 0 or len(yc) == 0:
        return float("nan"), float("nan")
    p1, p0 = yt.mean(), yc.mean()
    se = np.sqrt(p1 * (1 - p1) / len(yt) + p0 * (1 - p0) / len(yc))
    return float(p1 - p0), float(se)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outcome", default="visit")
    p.add_argument("--test-size", type=float, default=0.3)
    args = p.parse_args()

    spec = get_spec("hillstrom")
    df = load_raw(spec)
    X_all = build_features(df, spec)
    y_all = df[args.outcome].astype(int).to_numpy()
    arm_all = df[spec.treatment_col].to_numpy()

    rng = np.random.default_rng(0)
    is_test = rng.random(len(df)) < args.test_size
    print(f"\nMulti-treatment uplift — outcome: {args.outcome}")
    print(f"train {int((~is_test).sum()):,} / test {int(is_test.sum()):,}")
    print(f"arms: {ARMS} vs control {CONTROL!r}\n")

    # ---- One uplift model per arm, each against the shared control -------
    models, tau_test = {}, {}
    for arm in ARMS:
        keep = np.isin(arm_all, [arm, CONTROL])
        tr = keep & ~is_test
        t = (arm_all[tr] == arm).astype(int)
        model = SLearner().fit(X_all[tr], t, y_all[tr])
        models[arm] = model
        tau_test[arm] = model.predict_uplift(X_all[is_test])

        obs, se = observed_effect(y_all[keep & is_test], (arm_all[keep & is_test] == arm).astype(int))
        curve = qini_curve(
            y_all[keep & is_test],
            (arm_all[keep & is_test] == arm).astype(int),
            model.predict_uplift(X_all[keep & is_test]),
        )
        print(f"  {arm:<16} blanket uplift {obs:+.4f} (SE {se:.4f})   Qini {qini_coefficient(curve):.4f}")

    tau = pd.DataFrame(tau_test)
    best_arm = tau.idxmax(axis=1).to_numpy()
    best_tau = tau.max(axis=1).to_numpy()

    print(f"\n{'='*88}\nWHICH OFFER THE POLICY PICKS\n{'='*88}")
    for arm in ARMS:
        share = float((best_arm == arm).mean())
        print(f"  {arm:<16} {share:6.1%} of customers")

    # ---- Does the assignment agree with what the data shows? -------------
    # Each customer is assigned an arm by the model; we can only observe those
    # who happened to be randomized into the arm the model chose for them, or
    # into control. That subset is still a valid randomized comparison, because
    # the assignment rule uses pre-treatment features only.
    print(f"\n{'='*88}\nOBSERVED EFFECT WHEN THE MODEL'S CHOICE MATCHED THE ARM ACTUALLY SENT\n{'='*88}")
    test_arm, test_y = arm_all[is_test], y_all[is_test]
    rows = []
    for arm in ARMS:
        chose = best_arm == arm
        m = chose & ((test_arm == arm) | (test_arm == CONTROL))
        obs, se = observed_effect(test_y[m], (test_arm[m] == arm).astype(int))
        # The counterfactual of interest: the same customers, sent the other arm.
        other = [a for a in ARMS if a != arm][0]
        m_other = chose & ((test_arm == other) | (test_arm == CONTROL))
        obs_o, se_o = observed_effect(test_y[m_other], (test_arm[m_other] == other).astype(int))
        rows.append(
            {
                "model_choice": arm,
                "n_customers": int(chose.sum()),
                "uplift_chosen_arm": obs,
                "se_chosen": se,
                "uplift_other_arm": obs_o,
                "se_other": se_o,
                "advantage": obs - obs_o,
            }
        )
        print(f"  chose {arm:<16} -> sending it: {obs:+.4f} (SE {se:.4f}) | "
              f"sending {other}: {obs_o:+.4f} (SE {se_o:.4f}) | advantage {obs-obs_o:+.4f}")

    # ---- Policy value vs. the best blanket campaign ----------------------
    print(f"\n{'='*88}\nPOLICY VALUE\n{'='*88}")
    blanket = {}
    for arm in ARMS:
        keep = np.isin(test_arm, [arm, CONTROL])
        blanket[arm], _ = observed_effect(test_y[keep], (test_arm[keep] == arm).astype(int))
    best_blanket = max(blanket, key=blanket.get)

    matched = np.array([test_arm[i] == best_arm[i] for i in range(len(test_arm))])
    control_mask = test_arm == CONTROL
    p_treated = test_y[matched].mean()
    p_control = test_y[control_mask].mean()
    policy_uplift = p_treated - p_control
    se_policy = np.sqrt(
        p_treated * (1 - p_treated) / matched.sum() + p_control * (1 - p_control) / control_mask.sum()
    )

    print(f"  Best blanket campaign  : {best_blanket} -> {blanket[best_blanket]:+.4f}")
    print(f"  Other blanket campaign : {[a for a in ARMS if a != best_blanket][0]} -> "
          f"{blanket[[a for a in ARMS if a != best_blanket][0]]:+.4f}")
    print(f"  Model's per-customer choice: {policy_uplift:+.4f} "
          f"(95% CI [{policy_uplift-1.96*se_policy:+.4f}, {policy_uplift+1.96*se_policy:+.4f}], "
          f"n_matched={int(matched.sum()):,})")
    gain = policy_uplift - blanket[best_blanket]
    print(f"  Gain over the best blanket campaign: {gain:+.4f} "
          f"({gain/blanket[best_blanket]*100:+.1f}%)")
    if abs(gain) < 1.96 * se_policy:
        print("\n  That gain is inside the noise band. The honest read: on this dataset the")
        print("  men's email is simply the stronger creative for almost everyone, and")
        print("  per-customer creative selection does not clearly beat just sending it.")

    # ---- Where the arms genuinely differ ---------------------------------
    print(f"\n{'='*88}\nWHERE THE TWO CREATIVES ACTUALLY DIVERGE\n{'='*88}")
    print("  Segment-level observed uplift per arm (this is the real finding):\n")
    seg_rows = []
    for label, mask_all in [
        ("mens-only buyer", (df["mens"] == 1) & (df["womens"] == 0)),
        ("womens-only buyer", (df["womens"] == 1) & (df["mens"] == 0)),
        ("buys both", (df["mens"] == 1) & (df["womens"] == 1)),
    ]:
        m = mask_all.to_numpy()
        entry = {"segment": label, "n": int(m.sum())}
        for arm in ARMS:
            keep = m & np.isin(arm_all, [arm, CONTROL])
            obs, se = observed_effect(y_all[keep], (arm_all[keep] == arm).astype(int))
            entry[arm] = obs
            entry[f"{arm}_se"] = se
        entry["difference"] = entry[ARMS[0]] - entry[ARMS[1]]
        seg_rows.append(entry)
        print(f"  {label:<20} n={entry['n']:>6,}  "
              f"{ARMS[0]}: {entry[ARMS[0]]:+.4f}  {ARMS[1]}: {entry[ARMS[1]]:+.4f}  "
              f"diff {entry['difference']:+.4f}")

    worst = min(seg_rows, key=lambda r: min(r[ARMS[0]], r[ARMS[1]]))
    weak_arm = ARMS[0] if worst[ARMS[0]] < worst[ARMS[1]] else ARMS[1]
    strong_arm = [a for a in ARMS if a != weak_arm][0]
    print(f"\n  Sending {weak_arm} to {worst['segment']}s returns {worst[weak_arm]:+.4f} — against")
    print(f"  {worst[strong_arm]:+.4f} for {strong_arm}. Roughly "
          f"{(1 - worst[weak_arm]/worst[strong_arm])*100:.0f}% of the available lift is lost")
    print("  purely by sending the wrong creative. No amount of better *who* targeting")
    print("  recovers that; it is a *what* decision.")

    # ---- Figure ----------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels = [r["segment"] for r in seg_rows]
    x = np.arange(len(labels))
    w = 0.36
    for i, arm in enumerate(ARMS):
        vals = [r[arm] for r in seg_rows]
        errs = [1.96 * r[f"{arm}_se"] for r in seg_rows]
        ax.bar(x + (i - 0.5) * w, vals, w, yerr=errs, capsize=4,
               label=arm, color=[PALETTE["uplift"], PALETTE["response"]][i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Observed uplift on visits")
    ax.set_title("Which creative works, by shopper type (95% CI)")
    ax.legend(frameon=False)
    fig_path = save_figure(fig, "multi_treatment.png")

    out = REPORTS / "hillstrom_multi_treatment.json"
    out.write_text(
        json.dumps(
            {
                "outcome": args.outcome,
                "arms": ARMS,
                "arm_choice_share": {a: float((best_arm == a).mean()) for a in ARMS},
                "blanket_uplift": blanket,
                "policy_uplift": float(policy_uplift),
                "policy_se": float(se_policy),
                "gain_over_best_blanket": float(gain),
                "matched_vs_other": rows,
                "segment_effects": seg_rows,
            },
            indent=2,
            default=float,
        )
    )
    print(f"\nWrote {out} and {fig_path}")


if __name__ == "__main__":
    main()
