import numpy as np
import pandas as pd

from data_health_ftw.ml.baseline import (
    cross_validate_grouped,
    grouped_permutation_importance,
    make_models,
)


def make_frame(n_countries=30, years=(2010, 2015, 2019), seed=0):
    """Target depends strongly on `signal`; `noise` is irrelevant; a few values are missing."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_countries):
        country_level = rng.normal(0, 1)
        for year in years:
            signal = country_level + rng.normal(0, 0.3)
            rows.append({
                "country_code": f"C{i:02d}",
                "region": f"R{i % 6}",
                "year": year,
                "signal": signal,
                "noise": rng.normal(0, 1),
                "target": float(np.exp(3 + 0.8 * signal)),
            })
    frame = pd.DataFrame(rows)
    frame.loc[frame.sample(frac=0.1, random_state=1).index, "signal"] = np.nan
    return frame


def test_make_models_returns_the_three_models():
    assert set(make_models()) == {"mean_baseline", "ridge", "hist_gbm"}


def test_cross_validate_grouped_returns_one_row_per_model_with_metrics():
    frame = make_frame()
    result = cross_validate_grouped(
        frame, ["signal", "noise"], "target", "country_code", make_models(), n_splits=3
    )
    assert set(result["model"]) == {"mean_baseline", "ridge", "hist_gbm"}
    assert {"mae", "rmse", "r2", "mae_std", "r2_std", "target", "group_by"} <= set(result.columns)
    assert (result["mae"] >= 0).all()


def test_real_models_beat_the_mean_baseline_on_informative_data():
    frame = make_frame()
    result = cross_validate_grouped(
        frame, ["signal", "noise"], "target", "country_code", make_models(), n_splits=3
    ).set_index("model")
    assert result.loc["ridge", "mae"] < result.loc["mean_baseline", "mae"]
    assert result.loc["hist_gbm", "mae"] < result.loc["mean_baseline", "mae"]


def test_cross_validate_grouped_by_region_runs():
    frame = make_frame()
    result = cross_validate_grouped(
        frame, ["signal", "noise"], "target", "region", {"ridge": make_models()["ridge"]}, n_splits=3
    )
    assert result.loc[0, "group_by"] == "region"


def test_permutation_importance_ranks_the_informative_feature_first():
    frame = make_frame()
    importance = grouped_permutation_importance(
        frame, ["signal", "noise"], "target", "country_code",
        make_models()["hist_gbm"], n_splits=3, n_repeats=5,
    )
    assert importance.index[0] == "signal"
    assert importance["signal"] > importance["noise"]