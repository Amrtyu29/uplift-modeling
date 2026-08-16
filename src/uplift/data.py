"""Loading, encoding and splitting the randomized-experiment datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from .config import DATA_RAW, RANDOM_STATE, TEST_SIZE, DatasetSpec


@dataclass
class UpliftData:
    """A randomized experiment, split into train/test.

    ``t`` is binary treatment assignment, ``y`` the binary outcome, ``value``
    the observed monetary outcome (used only by the business simulation).
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    t_train: np.ndarray
    t_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    value_train: np.ndarray | None
    value_test: np.ndarray | None
    feature_names: list[str]
    spec: DatasetSpec

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)


def _read_dtypes(spec: DatasetSpec) -> dict[str, str]:
    """Column dtypes to apply at read time.

    Casting after the read defeats the purpose — the full-width frame has
    already been materialized by then, and that peak is what runs the machine
    out of memory on 14M rows. Binary columns go to int8 for the same reason.
    """
    dtypes: dict[str, str] = {c: spec.float_dtype for c in spec.numeric_features}
    for col in [spec.outcome_col, *spec.alt_outcome_cols]:
        dtypes[col] = "int8"
    if spec.treated_values is None:
        dtypes[spec.treatment_col] = "int8"
    return dtypes


def load_raw(
    spec: DatasetSpec,
    nrows: int | None = None,
    sample_frac: float | None = None,
    random_state: int = RANDOM_STATE,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Read a dataset, sampling during the read when asked to.

    Criteo is ~14M rows. Reading it whole and then calling ``.sample()`` needs
    the full frame in memory first, which is the step that actually hurts.
    Sampling chunk by chunk keeps peak memory proportional to the chunk, and —
    unlike ``nrows`` — it draws from the entire file rather than from whatever
    happens to sit at the top of it, which matters if the rows carry any order.
    """
    path = DATA_RAW / spec.filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/00_download_data.py --dataset {spec.name}`."
        )
    dtypes = _read_dtypes(spec)
    if sample_frac is None or sample_frac >= 1.0:
        return pd.read_csv(path, nrows=nrows, dtype=dtypes)

    rng = np.random.default_rng(random_state)
    parts = []
    for chunk in pd.read_csv(path, nrows=nrows, chunksize=chunksize, dtype=dtypes):
        parts.append(chunk.sample(frac=sample_frac, random_state=int(rng.integers(1e9))))
    return pd.concat(parts, ignore_index=True)


def encode_treatment(df: pd.DataFrame, spec: DatasetSpec) -> pd.Series:
    """Map the raw treatment column onto {0, 1}, dropping unused arms."""
    col = df[spec.treatment_col]
    if spec.treated_values is None:
        return col.astype(int)
    t = pd.Series(np.nan, index=df.index, dtype="float64")
    t[col.isin(spec.control_values or [])] = 0.0
    t[col.isin(spec.treated_values)] = 1.0
    return t


def build_features(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """One-hot the categoricals, pass the numerics through.

    No scaling: every model in this project is gradient-boosted trees, which are
    invariant to monotone transforms of the features.
    """
    X = df[spec.numeric_features].copy()
    if spec.categorical_features:
        dummies = pd.get_dummies(
            df[spec.categorical_features], prefix=spec.categorical_features, dtype=float
        )
        X = pd.concat([X, dummies], axis=1)
    return X.astype(spec.float_dtype)


def prepare(
    spec: DatasetSpec,
    outcome: str | None = None,
    nrows: int | None = None,
    sample_frac: float | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> UpliftData:
    """Load a dataset and split it, stratified on the treatment x outcome cell.

    Stratifying on the interaction keeps the treated/control ratio *and* the
    outcome base rate stable across the split, which matters because the Qini
    curve rescales control outcomes by the treated/control ratio.
    """
    df = load_raw(spec, nrows=nrows, sample_frac=sample_frac, random_state=random_state)

    t = encode_treatment(df, spec)
    keep = t.notna()
    df, t = df.loc[keep].reset_index(drop=True), t.loc[keep].reset_index(drop=True)

    outcome_col = outcome or spec.outcome_col
    y = df[outcome_col].astype(int).to_numpy()
    t = t.astype(int).to_numpy()
    X = build_features(df, spec)
    value = df[spec.revenue_col].to_numpy(dtype=float) if spec.revenue_col else None

    # Integer codes rather than concatenated strings: at 14M rows the string
    # version allocates 14M Python objects and costs more memory than the
    # feature matrix it is stratifying.
    strata = t * 2 + y
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=test_size, random_state=random_state, stratify=strata
    )

    return UpliftData(
        X_train=X.iloc[idx_train].reset_index(drop=True),
        X_test=X.iloc[idx_test].reset_index(drop=True),
        t_train=t[idx_train],
        t_test=t[idx_test],
        y_train=y[idx_train],
        y_test=y[idx_test],
        value_train=None if value is None else value[idx_train],
        value_test=None if value is None else value[idx_test],
        feature_names=list(X.columns),
        spec=spec,
    )


def randomization_check(X: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    """Covariate balance table — the sanity check the whole project rests on.

    If assignment really was randomized, no feature should differ meaningfully
    between arms. We report the standardized mean difference (SMD), because with
    64K rows a t-test will flag differences far too small to matter; the usual
    field convention is that |SMD| < 0.1 is balanced.
    """
    rows = []
    for col in X.columns:
        a, b = X.loc[t == 1, col].to_numpy(), X.loc[t == 0, col].to_numpy()
        pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd = 0.0 if pooled_sd == 0 else (a.mean() - b.mean()) / pooled_sd
        _, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append(
            {
                "feature": col,
                "mean_treated": a.mean(),
                "mean_control": b.mean(),
                "smd": smd,
                "p_value": p,
                "balanced": abs(smd) < 0.1,
            }
        )
    return pd.DataFrame(rows).sort_values("smd", key=np.abs, ascending=False)


def average_treatment_effect(y: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """ATE with a normal-approximation 95% CI on the difference of proportions."""
    y1, y0 = y[t == 1], y[t == 0]
    n1, n0 = len(y1), len(y0)
    p1, p0 = y1.mean(), y0.mean()
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    return {
        "rate_treated": float(p1),
        "rate_control": float(p0),
        "ate": float(p1 - p0),
        "ate_ci_low": float(p1 - p0 - 1.96 * se),
        "ate_ci_high": float(p1 - p0 + 1.96 * se),
        "relative_lift": float((p1 - p0) / p0) if p0 > 0 else float("nan"),
        "n_treated": int(n1),
        "n_control": int(n0),
    }
