import os
import sys

repo_root = os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(repo_root, "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_health_ftw.common import config
from data_health_ftw.ml.dataset import build_modeling_frame

RANDOM_STATE = 42
N_SPLITS = 5

raw = spark.read.table(config.GOLD_ML_FEATURES_TABLE).toPandas()
print(f"rows={len(raw)}, countries={raw['country_code'].nunique()}, years={sorted(int(y) for y in raw['year'].unique())}")
assert not raw.duplicated(config.GOLD_ML_UNIQUE_KEY).any(), "duplicated (country_code, year) keys"

null_by_year = (
    raw.groupby("year")[config.INDICATOR_NAMES]
    .apply(lambda g: g.isna().mean())
    .mul(100)
    .round(1)
)
null_by_year.T

features = list(config.ML_FEATURES)
targets = list(config.ML_TARGETS)

frame = build_modeling_frame(
    raw,
    features=features,
    targets=targets,
    min_year=config.ML_MIN_YEAR,
    fill_from_past_columns=config.ML_FILL_FROM_PAST,
)
print(f"modelling rows={len(frame)}, countries={frame['country_code'].nunique()}")
frame[features].isna().mean().mul(100).round(1).sort_values(ascending=False)

fig, axes = plt.subplots(2, len(targets), figsize=(5 * len(targets), 7))
for j, target in enumerate(targets):
    axes[0, j].hist(frame[target], bins=40)
    axes[0, j].set_title(f"{target} (raw)")
    axes[1, j].hist(np.log1p(frame[target]), bins=40)
    axes[1, j].set_title(f"log1p({target})")
plt.tight_layout()
plt.show()

frame[targets].describe().round(1)

for target in targets:
    y = np.log1p(frame[target])
    total = ((y - y.mean()) ** 2).sum()
    by_country = ((frame.assign(y=y).groupby("country_code")["y"].transform("mean") - y.mean()) ** 2).sum()
    by_year = ((frame.assign(y=y).groupby("year")["y"].transform("mean") - y.mean()) ** 2).sum()
    print(f"{target:20s} country={by_country / total:5.1%}  year={by_year / total:5.1%}")

corr = frame[features + targets].corr(method="spearman").loc[features, targets].round(2)
display(corr)  # noqa: F821

fig, ax = plt.subplots(figsize=(4 + len(targets), 5))
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(targets)), targets, rotation=45, ha="right")
ax.set_yticks(range(len(features)), features)
fig.colorbar(im)
plt.tight_layout()
plt.show()

frame[features].corr(method="spearman").round(2)

def evaluate(model_factory, X, y_log, groups, n_splits=N_SPLITS):
    """Return mean/std of R2 (log scale) and MAE (original scale) across grouped folds."""
    r2_scores, mae_scores = [], []
    for train_idx, valid_idx in GroupKFold(n_splits=n_splits).split(X, y_log, groups):
        model = model_factory()
        model.fit(X.iloc[train_idx], y_log.iloc[train_idx])
        pred = model.predict(X.iloc[valid_idx])
        r2_scores.append(r2_score(y_log.iloc[valid_idx], pred))
        mae_scores.append(mean_absolute_error(np.expm1(y_log.iloc[valid_idx]), np.expm1(pred)))
    return {
        "r2_mean": np.mean(r2_scores),
        "r2_std": np.std(r2_scores),
        "mae_mean": np.mean(mae_scores),
        "mae_std": np.std(mae_scores),
    }


MODELS = {
    "dummy_mean": lambda: DummyRegressor(strategy="mean"),
    "ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV()),
    "random_forest": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE),
    ),
}

rows = []
for target in targets:
    y_log = np.log1p(frame[target])
    for group_name in ("country_code", "region"):
        n_splits = min(N_SPLITS, frame[group_name].nunique())
        for model_name, factory in MODELS.items():
            result = evaluate(factory, frame[features], y_log, frame[group_name], n_splits)
            rows.append({"target": target, "groups": group_name, "model": model_name, **result})

results = pd.DataFrame(rows).round(3)
display(results)  

frame_year = frame.assign(year_feature=frame["year"])
rows = []
for target in targets:
    y_log = np.log1p(frame[target])
    for label, cols in (("without_year", features), ("with_year", features + ["year_feature"])):
        result = evaluate(MODELS["random_forest"], frame_year[cols], y_log, frame_year["country_code"])
        rows.append({"target": target, "setup": label, **result})

display(pd.DataFrame(rows).round(3)) 

fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 4))
for ax, target in zip(np.atleast_1d(axes), targets):
    model = MODELS["random_forest"]()
    model.fit(frame[features], np.log1p(frame[target]))
    importances = pd.Series(model[-1].feature_importances_, index=features).sort_values()
    importances.plot.barh(ax=ax, title=f"{target}")
plt.tight_layout()
plt.show()