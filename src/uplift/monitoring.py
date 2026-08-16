"""Drift monitoring on the treatment-effect distribution.

A conventional model monitor watches the feature distribution and the predicted
outcome. Neither is sufficient here. Uplift models fail in a specific way:
the *effect* decays — the offer stops working, or works on a different group —
while features and conversion rates look untouched. Novelty wears off,
competitors copy the offer, the audience saturates.

So the monitored quantity is the distribution of predicted uplift itself, plus
the share of the population the policy would suppress. When ground truth
eventually arrives from a small always-on randomized holdout, ``realized_effect``
closes the loop by comparing predicted to observed.

Implemented with numpy rather than Evidently: the metric that matters here
(PSI on tau) is ten lines, and Evidently's dependency footprint is heavy for a
container that otherwise only needs LightGBM. ``evidently_report`` is provided
for the standard feature-drift view when the library is present.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

WATCH_PSI = 0.10
ALERT_PSI = 0.25


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10, epsilon: float = 1e-6
) -> float:
    """PSI between two distributions using reference quantile bins.

    Quantile edges (not equal-width) so the bins carry equal reference mass and
    the statistic is not dominated by the tails of a skewed uplift distribution.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_prop = np.histogram(reference, bins=edges)[0] / len(reference) + epsilon
    cur_prop = np.histogram(current, bins=edges)[0] / len(current) + epsilon
    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


@dataclass
class DriftResult:
    batch: str
    n: int
    mean_uplift: float
    std_uplift: float
    psi: float
    suppressed_share: float
    ref_mean: float
    ref_mean_low: float
    ref_mean_high: float
    status: str
    reasons: list[str]
    checked_at: str


class UpliftDriftMonitor:
    """Compares each incoming batch's uplift distribution to a frozen reference.

    The reference is the training-time predicted-uplift distribution. Each batch
    is judged against a tolerance band sized for that batch, so "the mean moved"
    means moved by more than a batch of that size would move on its own.
    """

    def __init__(
        self,
        reference_uplift: np.ndarray,
        suppression_threshold: float = 0.0,
        z: float = 1.96,
    ):
        self.reference = np.asarray(reference_uplift, float)
        self.suppression_threshold = suppression_threshold
        self.z = z
        self.ref_mean = float(self.reference.mean())
        self.ref_std = float(self.reference.std(ddof=1))
        self.ref_suppressed = float((self.reference < suppression_threshold).mean())
        self.history: list[DriftResult] = []

    def mean_band(self, n: int) -> tuple[float, float]:
        """Tolerance band for the mean of a batch of size ``n``.

        The band has to scale with the batch, not with the reference. Bootstrapping
        the reference mean over 45K training rows gives an interval a few
        ten-thousandths wide; a 2K-row batch has roughly five times that much
        sampling error on its own, so every ordinary batch would trip the alarm.
        The right null is "could this batch have been drawn from the reference
        distribution", which is ref_std / sqrt(n).
        """
        se = self.ref_std / np.sqrt(max(n, 1))
        return self.ref_mean - self.z * se, self.ref_mean + self.z * se

    def check(self, uplift: np.ndarray, batch: str = "batch") -> DriftResult:
        uplift = np.asarray(uplift, float)
        psi = population_stability_index(self.reference, uplift)
        mean = float(uplift.mean())
        suppressed = float((uplift < self.suppression_threshold).mean())
        band_low, band_high = self.mean_band(len(uplift))

        reasons = []
        if psi >= ALERT_PSI:
            reasons.append(f"PSI {psi:.3f} >= {ALERT_PSI} — uplift distribution has shifted materially")
        elif psi >= WATCH_PSI:
            reasons.append(f"PSI {psi:.3f} >= {WATCH_PSI} — uplift distribution drifting")
        if mean < band_low:
            reasons.append(
                f"mean uplift {mean:.4f} below reference band [{band_low:.4f}, "
                f"{band_high:.4f}] — effect may be decaying"
            )
        elif mean > band_high:
            reasons.append(f"mean uplift {mean:.4f} above reference band — verify data pipeline")
        if suppressed > self.ref_suppressed + 0.10:
            reasons.append(
                f"suppressed share {suppressed:.1%} vs. reference {self.ref_suppressed:.1%} — "
                "sleeping-dog population growing"
            )

        status = "OK"
        if reasons:
            status = "ALERT" if (psi >= ALERT_PSI or mean < band_low) else "WARN"

        result = DriftResult(
            batch=batch,
            n=len(uplift),
            mean_uplift=mean,
            std_uplift=float(uplift.std()),
            psi=psi,
            suppressed_share=suppressed,
            ref_mean=self.ref_mean,
            ref_mean_low=band_low,
            ref_mean_high=band_high,
            status=status,
            reasons=reasons,
            checked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.history.append(result)
        return result

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.history])

    def write_log(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for r in self.history:
                f.write(json.dumps(asdict(r)) + "\n")
        return path


def realized_effect(y: np.ndarray, t: np.ndarray, uplift: np.ndarray, top_frac: float = 0.3) -> dict:
    """Closes the loop when labels arrive from the always-on randomized holdout.

    Compares what the model predicted for the top slice against what that slice
    actually did. A widening gap is the signal that the model — not just the
    data — needs retraining.
    """
    y, t, uplift = np.asarray(y, float), np.asarray(t, int), np.asarray(uplift, float)
    k = max(1, int(top_frac * len(y)))
    top = np.argsort(-uplift)[:k]
    yt, yc = y[top][t[top] == 1], y[top][t[top] == 0]
    if len(yt) == 0 or len(yc) == 0:
        return {"status": "insufficient_data"}
    observed = float(yt.mean() - yc.mean())
    predicted = float(uplift[top].mean())
    return {
        "top_frac": top_frac,
        "predicted_uplift": predicted,
        "observed_uplift": observed,
        "calibration_ratio": observed / predicted if predicted else float("nan"),
        "n_treated": int(len(yt)),
        "n_control": int(len(yc)),
    }


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, path: Path) -> Path | None:
    """Optional feature-drift HTML report, skipped cleanly if Evidently is absent."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except Exception:
        return None
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(path))
    return path
