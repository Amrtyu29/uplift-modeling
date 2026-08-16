"""Paths, dataset schemas and economic assumptions used across the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _p in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.3


@dataclass(frozen=True)
class Economics:
    """Unit economics for the business simulation.

    These are assumptions, not measurements. Every number the simulator reports
    is only as good as these three values, so they live in one place and are
    printed in every report.
    """

    value_per_conversion: float
    cost_per_contact: float
    currency: str = "$"


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the pipeline needs to know about a dataset.

    The rest of the code is dataset-agnostic: it only ever sees ``X`` (features),
    ``t`` (binary treatment) and ``y`` (binary outcome).
    """

    name: str
    filename: str
    treatment_col: str
    outcome_col: str
    numeric_features: list[str]
    categorical_features: list[str]
    economics: Economics
    # Hillstrom encodes treatment as a 3-level string; Criteo already has 0/1.
    treated_values: list[str] | None = None
    control_values: list[str] | None = None
    revenue_col: str | None = None
    alt_outcome_cols: list[str] = field(default_factory=list)
    drop_cols: list[str] = field(default_factory=list)
    # float32 halves the feature matrix. Irrelevant on 64K rows, decisive on
    # 14M: float64 puts the full Criteo frame plus its train/test copies past
    # the memory on a typical laptop. LightGBM bins features before splitting,
    # so the lost precision does not reach the model.
    float_dtype: str = "float64"


HILLSTROM = DatasetSpec(
    name="hillstrom",
    filename="hillstrom.csv",
    treatment_col="segment",
    treated_values=["Mens E-Mail", "Womens E-Mail"],
    control_values=["No E-Mail"],
    # `visit` is the primary outcome: it is the behaviour the email is actually
    # trying to cause, and at ~15% base rate it carries enough signal to
    # estimate heterogeneous effects on 64K rows. `conversion` (~0.9%) is
    # reported alongside it but is too sparse to segment on reliably.
    outcome_col="visit",
    alt_outcome_cols=["conversion"],
    revenue_col="spend",
    numeric_features=["recency", "history", "mens", "womens", "newbie"],
    categorical_features=["zip_code", "channel"],
    # `history_segment` is a binned copy of `history`; keeping both just gives
    # the trees a duplicated split with no new information.
    drop_cols=["history_segment"],
    economics=Economics(value_per_conversion=100.0, cost_per_contact=0.10),
)

CRITEO = DatasetSpec(
    name="criteo",
    filename="criteo-uplift-v2.1.csv.gz",
    treatment_col="treatment",
    outcome_col="visit",
    alt_outcome_cols=["conversion"],
    numeric_features=[f"f{i}" for i in range(12)],
    categorical_features=[],
    drop_cols=["exposure"],
    float_dtype="float32",
    economics=Economics(value_per_conversion=25.0, cost_per_contact=0.01),
)

DATASETS: dict[str, DatasetSpec] = {"hillstrom": HILLSTROM, "criteo": CRITEO}


def get_spec(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name]
