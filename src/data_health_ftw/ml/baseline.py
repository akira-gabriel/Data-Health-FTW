from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def _log_target(regressor: BaseEstimator) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def make_models(random_state: int = 42) -> dict[str, BaseEstimator]:
    return {
        "mean_baseline": _log_target(DummyRegressor(strategy="mean")),
        "ridge": _log_target(
            make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                Ridge(alpha=1.0),
            )
        ),
        "hist_gbm": _log_target(
            HistGradientBoostingRegressor(
                max_depth=4, learning_rate=0.05, max_iter=300, random_state=random_state
            )
        ),
    }


def cross_validate_grouped(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target: str,
    group_column: str,
    models: dict[str, BaseEstimator],
    n_splits: int = 5,
) -> pd.DataFrame:
    x, y, groups = frame[list(feature_columns)], frame[target], frame[group_column]
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model, x, y, groups=groups, cv=GroupKFold(n_splits=n_splits), scoring=SCORING
        )
        rows.append(
            {
                "model": name,
                "target": target,
                "group_by": group_column,
                "mae": -scores["test_mae"].mean(),
                "mae_std": scores["test_mae"].std(),
                "rmse": -scores["test_rmse"].mean(),
                "r2": scores["test_r2"].mean(),
                "r2_std": scores["test_r2"].std(),
            }
        )
    return pd.DataFrame(rows)


def grouped_permutation_importance(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target: str,
    group_column: str,
    estimator: BaseEstimator,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.Series:
    x, y, groups = frame[list(feature_columns)], frame[target], frame[group_column]
    totals = pd.Series(0.0, index=list(feature_columns))
    splits = list(GroupKFold(n_splits=n_splits).split(x, y, groups))
    for train_idx, test_idx in splits:
        fitted = estimator.fit(x.iloc[train_idx], y.iloc[train_idx])
        result = permutation_importance(
            fitted,
            x.iloc[test_idx],
            y.iloc[test_idx],
            scoring="neg_mean_absolute_error",
            n_repeats=n_repeats,
            random_state=random_state,
        )
        totals += pd.Series(result.importances_mean, index=totals.index)
    return (totals / len(splits)).sort_values(ascending=False)
