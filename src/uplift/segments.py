"""The four-box segmentation: persuadables, sure things, lost causes, sleeping dogs.

Uplift alone does not distinguish "won't convert either way" from "converts
either way" — both have tau ≈ 0. Separating them needs the *level* of the
control-arm outcome as well as the difference, which is why this module wants
both arm probabilities rather than just the uplift score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERSUADABLE = "persuadable"
SURE_THING = "sure_thing"
LOST_CAUSE = "lost_cause"
SLEEPING_DOG = "sleeping_dog"

SEGMENT_ORDER = [PERSUADABLE, SLEEPING_DOG, SURE_THING, LOST_CAUSE]

ACTIONS = {
    PERSUADABLE: "TARGET — the only group the spend actually moves",
    SLEEPING_DOG: "SUPPRESS — contact appears to reduce response",
    SURE_THING: "SKIP — converts without the offer; contacting it just costs money",
    LOST_CAUSE: "SKIP — does not respond either way",
}


def classify(
    uplift: np.ndarray,
    p_control: np.ndarray,
    uplift_cut: float,
    baseline_cut: float,
    sleeping_dog_threshold: float = 0.0,
) -> np.ndarray:
    """Assign each unit to one of the four boxes, given fixed cut-offs.

    A caveat worth stating plainly, because it is the thing interviewers push
    on: the four types are defined by a pair of potential outcomes, and no
    individual ever reveals both. These labels are therefore *not* recovered
    latent types — they are a decision rule over two estimated quantities, the
    predicted effect and the predicted do-nothing rate. What can be checked, and
    is checked in ``summarize``, is whether each labelled group behaves as
    claimed *on average* on randomized holdout data. That is the strongest
    available claim, and it is enough to act on.

    The cuts are passed in rather than derived here so that training and serving
    use identical boundaries — a customer's label must not depend on who else
    happened to be in the same scoring batch.

    - ``uplift >= uplift_cut``                       -> persuadable
    - ``uplift <  sleeping_dog_threshold``           -> sleeping dog
    - otherwise, split on the do-nothing rate: above ``baseline_cut`` the
      customer converts without help (sure thing), below it they do not convert
      either way (lost cause).
    """
    uplift, p_control = np.asarray(uplift), np.asarray(p_control)
    out = np.where(p_control >= baseline_cut, SURE_THING, LOST_CAUSE).astype(object)
    out[uplift >= uplift_cut] = PERSUADABLE
    out[uplift < sleeping_dog_threshold] = SLEEPING_DOG
    return out


def derive_cuts(
    uplift: np.ndarray,
    p_control: np.ndarray,
    uplift_quantile: float = 0.5,
    baseline_quantile: float = 0.5,
) -> dict[str, float]:
    """Choose segment boundaries from the training distribution.

    Quantiles rather than absolute thresholds: an absolute cut like "uplift >
    0.01" means something completely different on a 15% base rate than on a
    0.9% one, and would silently produce a 98%-persuadable or 0%-persuadable
    segmentation when moved between datasets.
    """
    return {
        "uplift_cut": float(np.quantile(uplift, uplift_quantile)),
        "baseline_cut": float(np.quantile(p_control, baseline_quantile)),
    }


def summarize(
    segments: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    value: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-segment observed behaviour on the randomized holdout.

    The ``observed_uplift`` column is the claim being checked: a segment called
    "persuadable" should show a positive treated-minus-control gap, and a
    segment called "sleeping dog" a negative one. Predicted labels that do not
    reproduce on held-out randomized data are just clustering.
    """
    y, t = np.asarray(y, float), np.asarray(t, int)
    rows = []
    for seg in SEGMENT_ORDER:
        m = segments == seg
        if not m.any():
            continue
        yt, yc = y[m & (t == 1)], y[m & (t == 0)]
        p1 = yt.mean() if len(yt) else np.nan
        p0 = yc.mean() if len(yc) else np.nan
        se = np.sqrt(
            (p1 * (1 - p1) / len(yt) if len(yt) else np.nan)
            + (p0 * (1 - p0) / len(yc) if len(yc) else np.nan)
        )
        row = {
            "segment": seg,
            "n": int(m.sum()),
            "share": float(m.mean()),
            "rate_treated": float(p1),
            "rate_control": float(p0),
            "observed_uplift": float(p1 - p0),
            "ci_low": float(p1 - p0 - 1.96 * se),
            "ci_high": float(p1 - p0 + 1.96 * se),
            "significant": bool(abs(p1 - p0) > 1.96 * se),
            "action": ACTIONS[seg],
        }
        if value is not None:
            v = np.asarray(value, float)
            vt, vc = v[m & (t == 1)], v[m & (t == 0)]
            row["value_per_user_treated"] = float(vt.mean()) if len(vt) else np.nan
            row["value_per_user_control"] = float(vc.mean()) if len(vc) else np.nan
            row["incremental_value_per_user"] = row["value_per_user_treated"] - row[
                "value_per_user_control"
            ]
        rows.append(row)
    return pd.DataFrame(rows)
