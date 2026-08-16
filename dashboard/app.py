"""Streamlit dashboard: explore the uplift model and the policy it implies.

    streamlit run dashboard/app.py

Four tabs, each answering one question a stakeholder actually asks:
  Overview  — does the model rank treatment effects at all?
  Policy    — how many people should we contact, and what is it worth?
  Segments  — who are these people, and does the labelling hold up?
  Score     — what would you do for this specific customer?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uplift.config import DATASETS, Economics, get_spec  # noqa: E402
from uplift.data import prepare  # noqa: E402
from uplift.evaluate import decile_table, evaluate  # noqa: E402
from uplift.pipeline import UpliftBundle, fit_response_model, load_default_bundle  # noqa: E402
from uplift.segments import ACTIONS, SEGMENT_ORDER, summarize  # noqa: E402
from uplift.simulate import break_even_contact_cost, compare_policies, headline  # noqa: E402

st.set_page_config(page_title="Uplift & Incremental Impact", page_icon="📈", layout="wide")

COLORS = {
    "uplift": "#1b6ca8",
    "response_model": "#e07b39",
    "random": "#9aa0a6",
    "treat_all": "#5a5a5a",
}


@st.cache_resource(show_spinner="Loading model...")
def load_model(dataset: str) -> UpliftBundle:
    return load_default_bundle(dataset)


@st.cache_data(show_spinner="Loading holdout data...")
def load_holdout(dataset: str, sample_frac: float | None):
    spec = get_spec(dataset)
    data = prepare(spec, sample_frac=sample_frac)
    return data


@st.cache_data(show_spinner="Scoring holdout...")
def score_holdout(dataset: str, sample_frac: float | None):
    """Score the holdout once and reuse it across tabs.

    Everything on the page is derived from this one frame, so the numbers in the
    curve, the policy table and the segment table cannot disagree with each other.
    """
    bundle = load_model(dataset)
    data = load_holdout(dataset, sample_frac)
    preds = bundle.predict(data.X_test)
    response = fit_response_model(data).predict_proba(data.X_test)[:, 1]
    return preds, response


def sidebar() -> tuple[str, Economics, float | None]:
    st.sidebar.title("Uplift Modeling")
    st.sidebar.caption("Who changes behaviour *because* of the offer")

    available = [d for d in DATASETS if list((ROOT / "models").glob(f"{d}_*.joblib"))]
    if not available:
        st.sidebar.error("No trained models found. Run `python scripts/02_train.py` first.")
        st.stop()
    dataset = st.sidebar.selectbox("Dataset", available)

    spec = get_spec(dataset)
    st.sidebar.subheader("Economics")
    st.sidebar.caption("These are assumptions. Every number on this page moves with them.")
    value = st.sidebar.number_input(
        "Value per conversion", min_value=0.01, value=float(spec.economics.value_per_conversion), step=5.0
    )
    cost = st.sidebar.number_input(
        "Cost per contact", min_value=0.0, value=float(spec.economics.cost_per_contact), step=0.05, format="%.2f"
    )
    frac = None
    if dataset == "criteo":
        frac = st.sidebar.slider("Sample fraction", 0.01, 0.5, 0.05, 0.01,
                                 help="Criteo is ~14M rows; sample for responsiveness.")
    return dataset, Economics(value, cost, spec.economics.currency), frac


def overview_tab(bundle, data, preds, response, econ):
    tau = preds["uplift"].to_numpy()
    res = evaluate(data.y_test, data.t_test, tau, name=bundle.model_name, n_boot=100)
    m = res["metrics"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Qini coefficient", f"{m['qini_coefficient']:.3f}",
              help="0 = no better than random targeting; 1 = perfect ranking of treatment effects.")
    c2.metric("95% CI", f"[{m['qini_ci_low']:.3f}, {m['qini_ci_high']:.3f}]")
    c3.metric("Model", bundle.model_name.replace("_", "-"))
    c4.metric("Holdout size", f"{data.n_test:,}")

    if m["qini_ci_low"] <= 0:
        st.warning(
            "The Qini confidence interval includes zero: on this holdout the ranking is not "
            "statistically distinguishable from random. Treat the policy numbers as indicative."
        )

    st.subheader("Qini curve")
    st.caption(
        "Cumulative incremental conversions as targeting deepens, with control outcomes "
        "rescaled to the treated group's size. Above the diagonal means the ranking is finding "
        "real effect heterogeneity."
    )
    curve = res["curve"]
    step = max(1, len(curve) // 400)
    c = curve.iloc[::step]
    fig = go.Figure()
    fig.add_scatter(x=c["fraction_targeted"] * 100, y=c["qini"], name="uplift model",
                    line=dict(color=COLORS["uplift"], width=3))
    fig.add_scatter(x=c["fraction_targeted"] * 100, y=c["random"], name="random targeting",
                    line=dict(color=COLORS["random"], dash="dash"))
    res_curve = evaluate(data.y_test, data.t_test, response, name="response", n_boot=0)["curve"].iloc[::step]
    fig.add_scatter(x=res_curve["fraction_targeted"] * 100, y=res_curve["qini"], name="response model",
                    line=dict(color=COLORS["response_model"], width=2))
    fig.update_layout(xaxis_title="% of customers targeted", yaxis_title="Cumulative incremental conversions",
                      height=420, hovermode="x unified", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Predicted uplift distribution")
        fig = go.Figure(go.Histogram(x=tau, nbinsx=60, marker_color=COLORS["uplift"]))
        fig.add_vline(x=0, line_color="black")
        fig.add_vline(x=float(tau.mean()), line_dash="dash", line_color=COLORS["response_model"],
                      annotation_text="mean")
        fig.update_layout(height=340, xaxis_title="Predicted uplift", yaxis_title="Customers",
                          margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        neg = float((tau < 0).mean())
        st.caption(
            f"{neg:.1%} of customers have a negative predicted effect (sleeping dogs)."
            if neg > 0 else
            "No customer has a negative predicted effect — this campaign helps everywhere; "
            "the heterogeneity is in magnitude, not sign."
        )

    with right:
        st.subheader("Calibration by decile")
        st.caption("Does predicted uplift actually track observed uplift on held-out randomized data?")
        dec = decile_table(data.y_test, data.t_test, tau)
        fig = go.Figure()
        fig.add_bar(x=dec["decile"], y=dec["observed_uplift"], name="observed",
                    marker_color=COLORS["uplift"],
                    error_y=dict(type="data", symmetric=False,
                                 array=dec["observed_ci_high"] - dec["observed_uplift"],
                                 arrayminus=dec["observed_uplift"] - dec["observed_ci_low"]))
        fig.add_scatter(x=dec["decile"], y=dec["predicted_uplift"], name="predicted",
                        line=dict(color=COLORS["response_model"], width=3))
        fig.update_layout(height=340, xaxis_title="Decile (1 = highest predicted uplift)",
                          yaxis_title="Uplift", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)


def policy_tab(bundle, data, preds, response, econ):
    st.subheader("What is the right campaign size?")
    basis = st.radio(
        "Revenue basis", ["observed", "modeled"], horizontal=True,
        help="observed = the dataset's own revenue column (no assumption about conversion value); "
             "modeled = incremental conversions priced at the assumed value.",
        index=0 if data.value_test is not None else 1,
    )
    if basis == "observed" and data.value_test is None:
        st.info("This dataset has no revenue column; using the modeled basis.")
        basis = "modeled"
    profit_col = "profit_observed" if basis == "observed" else "profit_modeled"

    comparison = compare_policies(
        {"uplift": preds["uplift"].to_numpy(), "response_model": response},
        data.y_test, data.t_test, econ, value=data.value_test,
    )
    h = headline(comparison, profit_col)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Optimal depth", f"top {h['best_depth']*100:.0f}%")
    c2.metric("Profit at optimum", f"{econ.currency}{h['profit']:,.0f}")
    c3.metric("vs. treat everyone", f"{h['gain_vs_treat_all_pct']:+.1f}%")
    c4.metric("vs. random", f"{h['gain_vs_random_pct']:+.1f}%")

    fig = go.Figure()
    for policy in ["uplift", "response_model", "random", "treat_all"]:
        grp = comparison[comparison["policy"] == policy].sort_values("depth")
        if not len(grp):
            continue
        fig.add_scatter(
            x=grp["depth"] * 100, y=grp[profit_col], name=policy.replace("_", " "),
            line=dict(color=COLORS[policy], width=3 if policy == "uplift" else 2,
                      dash="dash" if policy in ("random", "treat_all") else "solid"),
        )
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(height=430, xaxis_title="% of customers contacted",
                      yaxis_title=f"Incremental profit ({econ.currency})",
                      hovermode="x unified", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("When is selective targeting worth it?")
    st.caption(
        "Cheap channels favour contacting everyone — that is arithmetic, not a modelling failure. "
        "The sweep below finds the contact cost at which the uplift policy starts to win, using the "
        "bootstrap lower bound rather than the point estimate."
    )
    sweep = break_even_contact_cost(
        preds["uplift"].to_numpy(), data.y_test, data.t_test, econ,
        value=data.value_test, basis=basis, n_boot=100,
    )
    wins = sweep[sweep["uplift_wins"]]
    fig = go.Figure()
    fig.add_scatter(x=sweep["cost_per_contact"], y=sweep["profit_uplift"], name="uplift policy",
                    line=dict(color=COLORS["uplift"], width=3))
    fig.add_scatter(x=sweep["cost_per_contact"], y=sweep["profit_treat_all"], name="treat everyone",
                    line=dict(color=COLORS["treat_all"], dash="dash"))
    if len(wins):
        fig.add_vline(x=float(wins["cost_per_contact"].min()), line_dash="dot",
                      line_color=COLORS["response_model"], annotation_text="break-even")
    fig.add_vline(x=econ.cost_per_contact, line_dash="dashdot", line_color="#666",
                  annotation_text="your assumption")
    fig.update_layout(height=380, xaxis_title=f"Cost per contact ({econ.currency})",
                      yaxis_title=f"Profit ({econ.currency})", hovermode="x unified", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    if len(wins):
        be = float(wins["cost_per_contact"].min())
        if econ.cost_per_contact < be:
            st.info(
                f"At {econ.currency}{econ.cost_per_contact:.2f} per contact, broad targeting is close to "
                f"optimal. Selective targeting starts paying above {econ.currency}{be:.2f} per contact."
            )
        else:
            st.success(
                f"At {econ.currency}{econ.cost_per_contact:.2f} per contact you are above the "
                f"{econ.currency}{be:.2f} break-even: targeting the top {h['best_depth']*100:.0f}% is worth "
                f"{econ.currency}{h['profit'] - h['profit_treat_all']:,.0f} more than contacting everyone."
            )
    else:
        st.warning("Treating everyone is not beaten at any cost in this sweep, once noise is accounted for.")

    with st.expander("Profit by depth (table)"):
        pivot = comparison.pivot_table(index="depth", columns="policy", values=profit_col)
        st.dataframe(pivot.style.format("{:,.0f}"), use_container_width=True)


def segments_tab(bundle, data, preds, econ):
    st.subheader("The four boxes")
    st.caption(
        "These labels are a decision rule over two estimated quantities, not recovered ground truth — "
        "no customer ever reveals both potential outcomes. What can be checked is whether each group "
        "behaves as labelled on randomized holdout data, which is what the right-hand chart shows."
    )
    segs = preds["segment"].to_numpy()
    summary = summarize(segs, data.y_test, data.t_test, data.value_test)

    cols = st.columns(len(summary))
    for col, (_, row) in zip(cols, summary.iterrows()):
        col.metric(row["segment"].replace("_", " ").title(), f"{row['share']:.1%}",
                   f"{row['observed_uplift']:+.4f} observed")
        col.caption(ACTIONS[row["segment"]])

    left, right = st.columns(2)
    order = [s for s in SEGMENT_ORDER if s in set(summary["segment"])]
    summary = summary.set_index("segment").loc[order].reset_index()
    with left:
        fig = go.Figure(go.Bar(x=summary["segment"], y=summary["share"] * 100,
                               marker_color=["#1b6ca8", "#c0392b", "#3f9e5a", "#9aa0a6"][: len(summary)]))
        fig.update_layout(title="Share of customers (%)", height=360, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = go.Figure(go.Bar(
            x=summary["segment"], y=summary["observed_uplift"],
            marker_color=["#1b6ca8", "#c0392b", "#3f9e5a", "#9aa0a6"][: len(summary)],
            error_y=dict(type="data", symmetric=False,
                         array=summary["ci_high"] - summary["observed_uplift"],
                         arrayminus=summary["observed_uplift"] - summary["ci_low"]),
        ))
        fig.add_hline(y=0, line_color="black")
        fig.update_layout(title="Observed uplift on holdout (95% CI)", height=360, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        summary[["segment", "n", "share", "rate_treated", "rate_control", "observed_uplift",
                 "ci_low", "ci_high", "significant", "action"]]
        .style.format({"share": "{:.1%}", "rate_treated": "{:.4f}", "rate_control": "{:.4f}",
                       "observed_uplift": "{:+.4f}", "ci_low": "{:+.4f}", "ci_high": "{:+.4f}"}),
        use_container_width=True,
    )

    st.subheader("Who is in each segment?")
    seg_pick = st.selectbox("Segment", order)
    mask = segs == seg_pick
    profile = pd.DataFrame({
        "feature": data.X_test.columns,
        "segment mean": data.X_test[mask].mean().to_numpy(),
        "everyone else": data.X_test[~mask].mean().to_numpy(),
    })
    profile["difference"] = profile["segment mean"] - profile["everyone else"]
    st.dataframe(
        profile.sort_values("difference", key=np.abs, ascending=False)
        .style.format({"segment mean": "{:.3f}", "everyone else": "{:.3f}", "difference": "{:+.3f}"}),
        use_container_width=True, hide_index=True,
    )


def score_tab(bundle, econ):
    st.subheader("Score a single customer")
    st.caption("The same code path the API serves — enter a profile and see the decision and its reasoning.")

    if bundle.dataset != "hillstrom":
        st.info(f"The form is built for the Hillstrom schema; the loaded model is `{bundle.dataset}`.")
        return

    c1, c2, c3 = st.columns(3)
    recency = c1.slider("Months since last purchase", 1, 12, 3)
    history = c2.number_input("Spend in past year ($)", 0.0, 5000.0, 250.0, 25.0)
    newbie = c3.selectbox("New customer?", [0, 1], format_func=lambda v: "Yes" if v else "No")
    mens = c1.checkbox("Bought men's merchandise", value=True)
    womens = c2.checkbox("Bought women's merchandise", value=False)
    zip_code = c3.selectbox("Zip code type", ["Urban", "Surburban", "Rural"])
    channel = c1.selectbox("Channel", ["Web", "Phone", "Multichannel"])

    row = pd.DataFrame([{
        "recency": recency, "history": history, "mens": int(mens), "womens": int(womens),
        "newbie": int(newbie),
        "zip_code_Rural": float(zip_code == "Rural"),
        "zip_code_Surburban": float(zip_code == "Surburban"),
        "zip_code_Urban": float(zip_code == "Urban"),
        "channel_Multichannel": float(channel == "Multichannel"),
        "channel_Phone": float(channel == "Phone"),
        "channel_Web": float(channel == "Web"),
    }])
    p = bundle.predict(row).iloc[0]
    ev = p["uplift"] * econ.value_per_conversion - econ.cost_per_contact

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted uplift", f"{p['uplift']:+.4f}")
    m2.metric("If contacted", f"{p['p_treated']:.1%}")
    m3.metric("If left alone", f"{p['p_control']:.1%}")
    m4.metric("Expected value", f"{econ.currency}{ev:+.3f}")

    if ev > 0:
        st.success(f"**Contact.** Segment: {p['segment'].replace('_',' ')}. {ACTIONS[p['segment']]}  \n"
                   f"Expected gain of {econ.currency}{ev:.3f} per contact at "
                   f"{econ.currency}{econ.value_per_conversion:g} per conversion and "
                   f"{econ.currency}{econ.cost_per_contact:g} per contact.")
    else:
        st.warning(f"**Hold.** Segment: {p['segment'].replace('_',' ')}. {ACTIONS[p['segment']]}  \n"
                   f"The predicted effect of {p['uplift']:.4f} does not cover "
                   f"{econ.currency}{econ.cost_per_contact:g} per contact.")


def main() -> None:
    dataset, econ, frac = sidebar()
    bundle = load_model(dataset)
    data = load_holdout(dataset, frac)
    preds, response = score_holdout(dataset, frac)

    st.title("Uplift Modeling & Incremental Impact Simulator")
    st.caption(
        f"Model `{bundle.model_name}` on `{bundle.dataset}`, outcome `{bundle.outcome}`, "
        f"trained {bundle.trained_at}. All figures are computed on a randomized holdout of "
        f"{data.n_test:,} customers."
    )

    t1, t2, t3, t4 = st.tabs(["Overview", "Policy", "Segments", "Score a customer"])
    with t1:
        overview_tab(bundle, data, preds, response, econ)
    with t2:
        policy_tab(bundle, data, preds, response, econ)
    with t3:
        segments_tab(bundle, data, preds, econ)
    with t4:
        score_tab(bundle, econ)

    st.sidebar.divider()
    st.sidebar.caption(
        "Validation rests on the treatment being randomly assigned in the source data. "
        "Before trusting these numbers in production, confirm the policy with a live A/B test."
    )


if __name__ == "__main__":
    main()
