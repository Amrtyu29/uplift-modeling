"""Training orchestration and the serialized artifact the API and dashboard load."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from .config import MODELS, RANDOM_STATE, DatasetSpec, get_spec
from .data import UpliftData
from .evaluate import auuc, evaluate, qini_coefficient, qini_curve
from .learners import DEFAULT_PARAMS, BaseLearner, build_learners
from .segments import classify, derive_cuts
ARTIFACT_VERSION = "1.0.0"


@dataclass
class UpliftBundle:
    """Everything needed to score a customer, in one picklable object.

    ``baseline_model`` is fitted on the control arm only, so it estimates the
    do-nothing conversion probability. Uplift alone cannot tell a sure thing
    from a lost cause; pairing tau with this baseline is what makes the
    four-box segmentation possible at serving time, for any learner — including
    ones that expose no per-arm models of their own.
    """

    learner: BaseLearner
    baseline_model: Any
    feature_names: list[str]
    dataset: str
    outcome: str
    model_name: str
    metrics: dict
    uplift_cut: float
    baseline_cut: float
    sleeping_dog_threshold: float
    reference_uplift: np.ndarray
    economics: dict
    trained_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    version: str = ARTIFACT_VERSION

    def align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reindex incoming features onto the training schema.

        One-hot columns absent from a single scoring request (a channel the
        caller did not use) become 0 rather than a KeyError, and unexpected
        extra columns are dropped instead of shifting the matrix.
        """
        return X.reindex(columns=self.feature_names, fill_value=0.0).astype(float)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self.align(X)
        tau = np.asarray(self.learner.predict_uplift(X), float)
        p_control = self.baseline_model.predict_proba(X)[:, 1]
        p_treated = np.clip(p_control + tau, 0.0, 1.0)
        # Cuts are frozen at training time, rather than re-derived from
        # whatever batch happens to arrive — otherwise a customer's segment
        # would depend on who else was scored in the same request.
        segs = classify(
            tau, p_control, self.uplift_cut, self.baseline_cut, self.sleeping_dog_threshold
        )
        return pd.DataFrame(
            {
                "uplift": tau,
                "p_treated": p_treated,
                "p_control": p_control,
                "segment": segs,
            }
        )

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or MODELS / f"{self.dataset}_{self.model_name}.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "UpliftBundle":
        return joblib.load(path)


def fit_baseline_model(data: UpliftData) -> LGBMClassifier:
    """Do-nothing response model: fitted on the control arm alone."""
    model = LGBMClassifier(**DEFAULT_PARAMS)
    model.fit(data.X_train[data.t_train == 0], data.y_train[data.t_train == 0])
    return model


def fit_response_model(data: UpliftData) -> LGBMClassifier:
    """The naive benchmark: 'who is most likely to convert if we contact them'.

    This is the model most teams actually ship. It is trained on treated units
    only and knows nothing about counterfactuals — including it makes the
    comparison in the write-up concrete instead of rhetorical.
    """
    model = LGBMClassifier(**DEFAULT_PARAMS)
    model.fit(data.X_train[data.t_train == 1], data.y_train[data.t_train == 1])
    return model


def train_all(
    data: UpliftData,
    include_causal_forest: bool = True,
    n_boot: int = 200,
    mlflow_experiment: str | None = None,
) -> dict:
    """Fit every learner, evaluate each on the holdout, return everything.

    Model selection is on the Qini coefficient, but the bootstrap CI is carried
    through so the write-up can say whether the winner is actually separated
    from the runner-up.
    """
    tracker = _MLflowTracker(mlflow_experiment)
    learners = build_learners(include_causal_forest)
    results: dict[str, dict] = {}

    for name, learner in learners.items():
        t0 = time.perf_counter()
        learner.fit(data.X_train, data.t_train, data.y_train)
        fit_seconds = time.perf_counter() - t0

        tau_test = learner.predict_uplift(data.X_test)
        res = evaluate(data.y_test, data.t_test, tau_test, name=name, n_boot=n_boot)
        res["metrics"]["fit_seconds"] = round(fit_seconds, 2)
        res["learner"] = learner
        res["uplift_test"] = tau_test
        res["uplift_train"] = learner.predict_uplift(data.X_train)
        results[name] = res
        tracker.log(name, res["metrics"], data)

    tracker.close()
    return results


def cross_val_qini(
    data: UpliftData,
    include_causal_forest: bool = True,
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Repeated stratified CV on the training set, scoring Qini out-of-fold.

    Selecting on a single holdout is not defensible here. The Qini bootstrap CIs
    on 19K rows are wide enough to overlap for every model, so whichever learner
    happens to win that one split is close to a coin flip. Repeated CV separates
    "this learner is better" from "this split was lucky", and the fold-level
    standard deviation is reported so the write-up can say which it is.

    The test set stays untouched: CV chooses the model, the holdout reports it.
    """
    from sklearn.model_selection import RepeatedStratifiedKFold

    strata = data.t_train * 2 + data.y_train  # integer codes; see data.prepare
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    rows = []
    for fold, (tr, va) in enumerate(cv.split(data.X_train, strata)):
        X_tr, X_va = data.X_train.iloc[tr], data.X_train.iloc[va]
        for name, learner in build_learners(include_causal_forest).items():
            learner.fit(X_tr, data.t_train[tr], data.y_train[tr])
            tau = learner.predict_uplift(X_va)
            curve = qini_curve(data.y_train[va], data.t_train[va], tau)
            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "qini_coefficient": qini_coefficient(curve),
                    "auuc": auuc(curve),
                }
            )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("model")
        .agg(
            qini_mean=("qini_coefficient", "mean"),
            qini_std=("qini_coefficient", "std"),
            qini_min=("qini_coefficient", "min"),
            qini_max=("qini_coefficient", "max"),
            auuc_mean=("auuc", "mean"),
            n_folds=("qini_coefficient", "size"),
        )
        .reset_index()
        .sort_values("qini_mean", ascending=False)
    )
    summary["qini_se"] = summary["qini_std"] / np.sqrt(summary["n_folds"])
    return summary


def select_best(results: dict, metric: str = "qini_coefficient") -> str:
    return max(results, key=lambda k: results[k]["metrics"][metric])


def select_best_cv(cv_summary: pd.DataFrame) -> tuple[str, bool]:
    """Winner by mean CV Qini, plus whether it clears the runner-up's noise.

    "Clears" means the winner's mean sits more than one combined standard error
    above the runner-up's. Anything tighter is a tie, and a tie should be broken
    on interpretability and serving cost rather than on a decimal place.
    """
    best, second = cv_summary.iloc[0], cv_summary.iloc[1]
    combined_se = float(np.hypot(best["qini_se"], second["qini_se"]))
    separated = bool(best["qini_mean"] - second["qini_mean"] > combined_se)
    return str(best["model"]), separated


def build_bundle(
    data: UpliftData,
    results: dict,
    model_name: str,
    spec: DatasetSpec | None = None,
) -> UpliftBundle:
    spec = spec or data.spec
    learner = results[model_name]["learner"]
    baseline = fit_baseline_model(data)
    tau_train = results[model_name]["uplift_train"]
    p_control_train = baseline.predict_proba(data.X_train)[:, 1]
    cuts = derive_cuts(tau_train, p_control_train)
    return UpliftBundle(
        learner=learner,
        baseline_model=baseline,
        feature_names=data.feature_names,
        dataset=spec.name,
        outcome=spec.outcome_col,
        model_name=model_name,
        metrics=results[model_name]["metrics"],
        uplift_cut=cuts["uplift_cut"],
        baseline_cut=cuts["baseline_cut"],
        sleeping_dog_threshold=0.0,
        reference_uplift=np.asarray(tau_train, float),
        economics={
            "value_per_conversion": spec.economics.value_per_conversion,
            "cost_per_contact": spec.economics.cost_per_contact,
            "currency": spec.economics.currency,
        },
    )


class _MLflowTracker:
    """Thin MLflow wrapper that degrades to a no-op when MLflow is absent.

    Experiment tracking should not be the reason a fresh clone fails to run.
    """

    def __init__(self, experiment: str | None):
        self.mlflow = None
        if not experiment:
            return
        try:
            import mlflow

            mlflow.set_tracking_uri(f"file://{(MODELS.parent / 'mlruns').as_posix()}")
            mlflow.set_experiment(experiment)
            self.mlflow = mlflow
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[mlflow] tracking disabled ({exc})")

    def log(self, name: str, metrics: dict, data: UpliftData) -> None:
        if self.mlflow is None:
            return
        with self.mlflow.start_run(run_name=name):
            self.mlflow.log_params(
                {**{f"lgbm_{k}": v for k, v in DEFAULT_PARAMS.items()}, "learner": name,
                 "n_train": data.n_train, "n_test": data.n_test, "dataset": data.spec.name}
            )
            self.mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

    def close(self) -> None:
        pass


def save_metrics(results: dict, path: Path, extra: dict | None = None) -> Path:
    """Flat metrics table plus context, written next to the figures."""
    rows = [r["metrics"] for r in results.values()]
    payload = {"models": rows, **(extra or {})}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


def load_default_bundle(dataset: str = "hillstrom") -> UpliftBundle:
    """Load the most recently trained bundle for a dataset.

    Sorting by name would pick alphabetically — so a stale `s_learner` artifact
    would keep being served after a rerun selected `x_learner`, silently, with
    no error anywhere. Modification time is what "the current model" means here.
    """
    spec = get_spec(dataset)
    candidates = sorted(MODELS.glob(f"{spec.name}_*.joblib"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No trained model for {dataset!r} in {MODELS}. Run `python scripts/02_train.py`."
        )
    return UpliftBundle.load(candidates[-1])


__all__ = [
    "UpliftBundle",
    "train_all",
    "select_best",
    "build_bundle",
    "fit_response_model",
    "fit_baseline_model",
    "save_metrics",
    "load_default_bundle",
    "RANDOM_STATE",
]
