"""FastAPI service: customer features in, uplift score and recommended action out.

    uvicorn api.main:app --reload

The endpoint returns a decision, not just a number. A raw uplift score is not
actionable on its own — whether to contact someone depends on the cost of the
contact and the value of a conversion, so the service applies those economics
and returns the resulting recommendation, with the reasoning attached.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uplift.pipeline import UpliftBundle, load_default_bundle  # noqa: E402
from uplift.segments import ACTIONS  # noqa: E402

DATASET = os.environ.get("UPLIFT_DATASET", "hillstrom")
MODEL_PATH = os.environ.get("UPLIFT_MODEL_PATH")

app = FastAPI(
    title="Uplift Modeling & Incremental Impact API",
    description=__doc__,
    version="1.0.0",
)

_bundle: UpliftBundle | None = None


def get_bundle() -> UpliftBundle:
    """Load the model once, on first use.

    Loading at import time would make the container fail to start when no model
    has been trained yet, and would break `--reload` during development.
    """
    global _bundle
    if _bundle is None:
        _bundle = UpliftBundle.load(Path(MODEL_PATH)) if MODEL_PATH else load_default_bundle(DATASET)
    return _bundle


class Customer(BaseModel):
    """A Hillstrom customer record, as the campaign system would send it."""

    recency: float = Field(..., ge=0, description="Months since last purchase")
    history: float = Field(..., ge=0, description="Dollars spent in the past year")
    mens: int = Field(0, ge=0, le=1, description="Bought men's merchandise in the past year")
    womens: int = Field(0, ge=0, le=1, description="Bought women's merchandise in the past year")
    newbie: int = Field(0, ge=0, le=1, description="New customer in the past year")
    zip_code: Literal["Rural", "Surburban", "Urban"] = "Urban"
    channel: Literal["Phone", "Web", "Multichannel"] = "Web"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "recency": 2,
                    "history": 420.50,
                    "mens": 1,
                    "womens": 0,
                    "newbie": 0,
                    "zip_code": "Urban",
                    "channel": "Web",
                }
            ]
        }
    }


class ScoreRequest(BaseModel):
    customers: list[Customer] = Field(..., min_length=1, max_length=10_000)
    cost_per_contact: float | None = Field(
        None, ge=0, description="Overrides the model's trained-in assumption"
    )
    value_per_conversion: float | None = Field(None, gt=0)


class Score(BaseModel):
    uplift: float = Field(..., description="Estimated change in P(visit) caused by contacting")
    p_treated: float
    p_control: float
    segment: str
    segment_meaning: str
    recommendation: Literal["contact", "hold"]
    expected_value: float = Field(..., description="uplift * value_per_conversion - cost_per_contact")
    reason: str


class ScoreResponse(BaseModel):
    scores: list[Score]
    model_name: str
    dataset: str
    outcome: str
    trained_at: str
    economics: dict
    n_recommended: int


def _to_frame(customers: list[Customer]) -> pd.DataFrame:
    """Encode requests the same way training did.

    ``UpliftBundle.align`` fills any one-hot column the request did not produce,
    so a batch that happens to contain only Urban customers still scores against
    the full trained schema.
    """
    raw = pd.DataFrame([c.model_dump() for c in customers])
    numeric = raw[["recency", "history", "mens", "womens", "newbie"]].astype(float)
    dummies = pd.get_dummies(raw[["zip_code", "channel"]], prefix=["zip_code", "channel"], dtype=float)
    return pd.concat([numeric, dummies], axis=1)


@app.get("/health")
def health() -> dict:
    """Liveness plus enough model identity to tell which artifact is serving."""
    try:
        b = get_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "model": b.model_name,
        "dataset": b.dataset,
        "outcome": b.outcome,
        "trained_at": b.trained_at,
        "version": b.version,
        "qini_coefficient": b.metrics.get("qini_coefficient"),
    }


@app.get("/model-info")
def model_info() -> dict:
    """Full model card: metrics, thresholds and the economics baked into it."""
    b = get_bundle()
    return {
        "model_name": b.model_name,
        "dataset": b.dataset,
        "outcome": b.outcome,
        "trained_at": b.trained_at,
        "version": b.version,
        "features": b.feature_names,
        "metrics": b.metrics,
        "segment_cuts": {
            "uplift_cut": b.uplift_cut,
            "baseline_cut": b.baseline_cut,
            "sleeping_dog_threshold": b.sleeping_dog_threshold,
        },
        "economics": b.economics,
        "reference_uplift": {
            "n": int(len(b.reference_uplift)),
            "mean": float(np.mean(b.reference_uplift)),
            "std": float(np.std(b.reference_uplift)),
        },
    }


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    """Score customers and recommend contact or hold for each.

    The recommendation is an expected-value test, not a segment lookup:
    contact when ``uplift * value_per_conversion > cost_per_contact``. Segments
    are for explaining the decision to a human, not for making it — a
    persuadable with a tiny effect and an expensive channel is still a hold.
    """
    b = get_bundle()
    value = request.value_per_conversion or b.economics["value_per_conversion"]
    cost = request.cost_per_contact if request.cost_per_contact is not None else b.economics["cost_per_contact"]
    currency = b.economics.get("currency", "$")

    preds = b.predict(_to_frame(request.customers))
    expected = preds["uplift"] * value - cost

    scores = []
    for i, row in preds.iterrows():
        ev = float(expected.iloc[i])
        contact = ev > 0
        if row["segment"] == "sleeping_dog":
            reason = "predicted negative effect — contacting is expected to reduce response"
        elif contact:
            reason = (
                f"expected gain {currency}{ev:.3f} per contact "
                f"({row['uplift']:.4f} x {currency}{value:g} > {currency}{cost:g})"
            )
        else:
            reason = (
                f"expected loss {currency}{ev:.3f} per contact — effect of {row['uplift']:.4f} "
                f"is too small to cover {currency}{cost:g}"
            )
        scores.append(
            Score(
                uplift=float(row["uplift"]),
                p_treated=float(row["p_treated"]),
                p_control=float(row["p_control"]),
                segment=row["segment"],
                segment_meaning=ACTIONS[row["segment"]],
                recommendation="contact" if contact else "hold",
                expected_value=ev,
                reason=reason,
            )
        )

    return ScoreResponse(
        scores=scores,
        model_name=b.model_name,
        dataset=b.dataset,
        outcome=b.outcome,
        trained_at=b.trained_at,
        economics={"value_per_conversion": value, "cost_per_contact": cost, "currency": currency},
        n_recommended=sum(s.recommendation == "contact" for s in scores),
    )


@app.post("/campaign")
def campaign(request: ScoreRequest, depth: float | None = None) -> dict:
    """Plan a campaign over a list: who to contact, and what it is worth.

    ``depth`` targets a fixed top fraction by uplift (a budget constraint).
    Omitting it lets expected value decide the size, which is the right default
    when the budget is not the binding constraint.
    """
    if depth is not None and not 0 < depth <= 1:
        raise HTTPException(status_code=422, detail="depth must be in (0, 1]")

    b = get_bundle()
    value = request.value_per_conversion or b.economics["value_per_conversion"]
    cost = request.cost_per_contact if request.cost_per_contact is not None else b.economics["cost_per_contact"]

    preds = b.predict(_to_frame(request.customers))
    n = len(preds)
    if depth is None:
        selected = (preds["uplift"] * value - cost > 0).to_numpy()
        rule = "expected value > 0"
    else:
        k = max(1, int(round(depth * n)))
        order = np.argsort(-preds["uplift"].to_numpy())
        selected = np.zeros(n, bool)
        selected[order[:k]] = True
        rule = f"top {depth:.0%} by uplift"

    chosen = preds[selected]
    return {
        "rule": rule,
        "n_customers": n,
        "n_contacted": int(selected.sum()),
        "contact_rate": float(selected.mean()),
        "expected_incremental_conversions": float(chosen["uplift"].sum()),
        "expected_cost": float(selected.sum() * cost),
        "expected_value": float(chosen["uplift"].sum() * value - selected.sum() * cost),
        "segments": preds.loc[selected, "segment"].value_counts().to_dict(),
        "selected_indices": np.flatnonzero(selected).tolist(),
        "economics": {"value_per_conversion": value, "cost_per_contact": cost},
        "caveat": (
            "Expected values assume the model is calibrated and the campaign effect "
            "still holds. Validate with a randomized holdout before trusting the total."
        ),
    }
