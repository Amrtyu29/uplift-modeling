"""Meta-learners for heterogeneous treatment effects.

Every learner exposes the same two-method interface::

    learner.fit(X, t, y)
    tau = learner.predict_uplift(X)   # estimated P(y=1 | do(t=1)) - P(y=1 | do(t=0))

They are implemented directly on top of LightGBM rather than pulled from
`causalml`, because the whole point of the exercise is that the S/T/X
distinction is a handful of lines of bookkeeping around ordinary classifiers —
and because writing the X-learner out makes its propensity weighting visible
instead of hidden behind a constructor argument.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import clone

from .config import RANDOM_STATE

# Deliberately much more heavily regularized than a normal classification
# setup. Uplift is a *difference* of two small probabilities, so variance in
# either arm's model lands directly on the estimate and does not cancel. A grid
# search over CV Qini (scripts/02a_tune.py) is unambiguous about this: going
# from num_leaves=31/min_child_samples=100 to the settings below roughly doubles
# CV Qini for every learner, even though the deeper trees score better on
# ordinary outcome AUC. Accuracy on y is not the objective here.
DEFAULT_PARAMS = dict(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=7,
    min_child_samples=800,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)


def _clf(**overrides) -> LGBMClassifier:
    return LGBMClassifier(**{**DEFAULT_PARAMS, **overrides})


def _reg(**overrides) -> LGBMRegressor:
    return LGBMRegressor(**{**DEFAULT_PARAMS, **overrides})


def _as_frame(X) -> pd.DataFrame:
    """Keep feature names attached all the way through fit and predict.

    LightGBM's sklearn wrapper stamps synthetic names ("Column_0", ...) onto a
    model fitted from a bare numpy array, then sklearn warns when predict is
    handed an array without them. Carrying a DataFrame throughout avoids the
    warning and, more usefully, makes a column-order mismatch between training
    and serving fail loudly instead of silently scoring the wrong features.
    """
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    X = np.asarray(X)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])


class BaseLearner(ABC):
    name: str = "base"

    @abstractmethod
    def fit(
        self, X, t: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "BaseLearner":
        """Fit on (X, t, y).

        ``sample_weight`` exists for the Bayesian bootstrap in `bayesian.py`,
        which draws a posterior by refitting under Dirichlet row weights. Every
        learner must honour it, including in whichever per-arm subset it builds
        internally — silently ignoring it would produce a posterior with no
        spread at all, which looks like a confident model rather than a bug.
        """

    @abstractmethod
    def predict_uplift(self, X) -> np.ndarray: ...

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"


class SLearner(BaseLearner):
    """One model on everyone, with treatment appended as a feature.

    Cheapest option and the most data-efficient, but it has a well-known failure
    mode: the treatment column is just one feature among many, so a regularized
    tree ensemble can simply decline to split on it and return uplift ≈ 0
    everywhere. Worth including precisely because that failure is instructive.
    """

    name = "s_learner"
    TREATMENT_COL = "__treatment__"

    def __init__(self, model=None):
        self.model = model or _clf()

    def fit(self, X, t, y, sample_weight=None):
        Xt = _as_frame(X).assign(**{self.TREATMENT_COL: np.asarray(t, dtype=float)})
        self.model.fit(Xt, y, sample_weight=sample_weight)
        return self

    def predict_uplift(self, X):
        Xf = _as_frame(X)
        ones = Xf.assign(**{self.TREATMENT_COL: 1.0})
        zeros = Xf.assign(**{self.TREATMENT_COL: 0.0})
        return self.model.predict_proba(ones)[:, 1] - self.model.predict_proba(zeros)[:, 1]


class TLearner(BaseLearner):
    """Two independent models, one per arm; uplift is their difference.

    Cannot ignore the treatment (it is baked into the data split) so it always
    produces non-trivial uplift. The cost is that each model sees only its own
    arm, so the smaller arm's model is noisier — and that noise does not cancel
    when you subtract, it adds.
    """

    name = "t_learner"

    def __init__(self, model_treated=None, model_control=None):
        self.model_treated = model_treated or _clf()
        self.model_control = model_control or _clf()

    def fit(self, X, t, y, sample_weight=None):
        Xf, t = _as_frame(X), np.asarray(t)
        w = None if sample_weight is None else np.asarray(sample_weight)
        self.model_treated.fit(
            Xf[t == 1], y[t == 1], sample_weight=None if w is None else w[t == 1]
        )
        self.model_control.fit(
            Xf[t == 0], y[t == 0], sample_weight=None if w is None else w[t == 0]
        )
        return self

    def predict_uplift(self, X):
        Xf = _as_frame(X)
        return (
            self.model_treated.predict_proba(Xf)[:, 1]
            - self.model_control.predict_proba(Xf)[:, 1]
        )


class XLearner(BaseLearner):
    """Two-stage learner that fixes the T-learner's imbalance problem.

    Stage 1: outcome models per arm (same as the T-learner).
    Stage 2: impute each unit's *individual* treatment effect using the other
        arm's model, then regress those imputed effects on X — separately per
        arm, giving tau_1 (fit on treated) and tau_0 (fit on control).
    Combine: tau(x) = g(x) * tau_0(x) + (1 - g(x)) * tau_1(x), with g the
        propensity score.

    The propensity weighting is the whole trick: where treated units are scarce,
    tau_1 is the noisy estimate and the weight (1-g) on it is small, so the
    combination leans on whichever arm actually has data at that point in
    feature space. With a 2:1 randomized split like Hillstrom's the correction
    is modest; under a 15:1 split like Criteo's it matters a great deal.
    """

    name = "x_learner"

    def __init__(self, outcome_model=None, effect_model=None, propensity_model=None):
        self.m1 = clone(outcome_model) if outcome_model is not None else _clf()
        self.m0 = clone(outcome_model) if outcome_model is not None else _clf()
        self.tau1 = clone(effect_model) if effect_model is not None else _reg()
        self.tau0 = clone(effect_model) if effect_model is not None else _reg()
        self.propensity_model = propensity_model or _clf(n_estimators=100)

    def fit(self, X, t, y, sample_weight=None):
        Xf, t, y = _as_frame(X), np.asarray(t), np.asarray(y)
        X1, y1 = Xf[t == 1], y[t == 1]
        X0, y0 = Xf[t == 0], y[t == 0]
        w = None if sample_weight is None else np.asarray(sample_weight)
        w1 = None if w is None else w[t == 1]
        w0 = None if w is None else w[t == 0]

        self.m1.fit(X1, y1, sample_weight=w1)
        self.m0.fit(X0, y0, sample_weight=w0)

        # Imputed treatment effects: for a treated unit, what the control model
        # says it *would* have done; for a control unit, the mirror image.
        d1 = y1 - self.m0.predict_proba(X1)[:, 1]
        d0 = self.m1.predict_proba(X0)[:, 1] - y0

        self.tau1.fit(X1, d1, sample_weight=w1)
        self.tau0.fit(X0, d0, sample_weight=w0)
        self.propensity_model.fit(Xf, t, sample_weight=w)
        return self

    def predict_uplift(self, X):
        Xf = _as_frame(X)
        # Clipped so a near-0/1 propensity cannot hand all the weight to the arm
        # that has essentially no data there.
        g = np.clip(self.propensity_model.predict_proba(Xf)[:, 1], 0.05, 0.95)
        return g * self.tau0.predict(Xf) + (1 - g) * self.tau1.predict(Xf)


class ClassTransformLearner(BaseLearner):
    """Single model on a relabelled target (Lai / "class transformation").

    Define z = 1 when (treated and converted) or (control and did not convert).
    Under a 50/50 randomized split, 2*P(z=1|x) - 1 is exactly the uplift. Off
    50/50 the estimate needs the propensity correction applied below.

    Included as the cheap, single-model benchmark: if it matches the fancier
    learners, the extra machinery is not earning its keep.
    """

    name = "class_transform"

    def __init__(self, model=None):
        self.model = model or _clf()

    def fit(self, X, t, y, sample_weight=None):
        Xf, t, y = _as_frame(X), np.asarray(t), np.asarray(y)
        z = ((t == 1) & (y == 1)) | ((t == 0) & (y == 0))
        w = None if sample_weight is None else np.asarray(sample_weight)
        # Weighted mean, so the transform's propensity correction matches the
        # reweighted sample the model actually saw.
        self.p_treat = float(np.average(t, weights=w))
        self.model.fit(Xf, z.astype(int), sample_weight=w)
        return self

    def predict_uplift(self, X):
        p_z = self.model.predict_proba(_as_frame(X))[:, 1]
        # Generalized transform: reduces to 2*p_z - 1 when p_treat = 0.5.
        return p_z / self.p_treat - (1 - p_z) / (1 - self.p_treat)


class CausalForestLearner(BaseLearner):
    """`econml`'s CausalForestDML, if it is installed.

    Kept optional on purpose: econml pins scientific-stack versions tightly and
    is the single most likely thing to break a fresh install. The pipeline
    skips it with a warning rather than failing.
    """

    name = "causal_forest"

    def __init__(self, **kwargs):
        from econml.dml import CausalForestDML  # imported lazily

        self.model = CausalForestDML(
            # model_y is a *regressor* even though the outcome is binary: DML
            # fits E[Y|X], a probability, and econml rejects a classifier there
            # outright. model_t stays a classifier because treatment is discrete.
            model_y=_reg(),
            model_t=_clf(),
            discrete_treatment=True,
            n_estimators=300,
            min_samples_leaf=50,
            max_depth=None,
            random_state=RANDOM_STATE,
            **kwargs,
        )

    @staticmethod
    def is_available() -> bool:
        try:
            import econml  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, X, t, y, sample_weight=None):
        # econml passes numpy between its own stages, and LightGBM stamps
        # synthetic feature names onto a model fitted from an array — so sklearn
        # warns about a mismatch that cannot happen here, since every array on
        # this path comes from the same frame in the same column order.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
            self.model.fit(
                Y=np.asarray(y),
                T=np.asarray(t),
                X=_as_frame(X).to_numpy(),
                sample_weight=None if sample_weight is None else np.asarray(sample_weight),
            )
        return self

    def predict_uplift(self, X):
        return np.asarray(self.model.effect(_as_frame(X).to_numpy())).ravel()


_CAUSAL_FOREST_NOTICE_SHOWN = False


def build_learners(include_causal_forest: bool = True) -> dict[str, BaseLearner]:
    """The model zoo, in the order they are reported.

    A missing econml drops the causal forest from the comparison rather than
    failing the run, but it says so once — a model quietly absent from a
    results table is worse than one that is loudly missing.
    """
    global _CAUSAL_FOREST_NOTICE_SHOWN
    learners: dict[str, BaseLearner] = {
        "s_learner": SLearner(),
        "t_learner": TLearner(),
        "x_learner": XLearner(),
        "class_transform": ClassTransformLearner(),
    }
    if include_causal_forest:
        if CausalForestLearner.is_available():
            learners["causal_forest"] = CausalForestLearner()
        elif not _CAUSAL_FOREST_NOTICE_SHOWN:
            # Printed once, not per CV fold.
            print("[learners] econml not installed — causal forest excluded from the comparison.")
            _CAUSAL_FOREST_NOTICE_SHOWN = True
    return learners
