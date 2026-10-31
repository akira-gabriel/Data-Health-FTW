import numpy as np
import pandas as pd
import pytest

from data_health_ftw.ml.dataset import build_modeling_frame, fill_from_past


def make_df(rows):
    return pd.DataFrame(rows, columns=["country_code", "region", "year", "smoking", "pm25", "target"])


def test_fill_from_past_uses_earlier_years_only():
    df = make_df([
        ("BRA", "AMR", 2015, 20.0, 1.0, 1.0),
        ("BRA", "AMR", 2019, np.nan, 1.0, 1.0),
        ("BRA", "AMR", 2020, 18.0, 1.0, 1.0),
    ])
    result = fill_from_past(df, ["smoking"])
    assert result["smoking"].tolist() == [20.0, 20.0, 18.0]  # 2019 tem como referência 2015, not 2020


def test_fill_from_past_does_not_leak_across_countries():
    df = make_df([
        ("ARG", "AMR", 2015, 30.0, 1.0, 1.0),
        ("BRA", "AMR", 2015, np.nan, 1.0, 1.0),
    ])
    result = fill_from_past(df, ["smoking"]).set_index("country_code")
    assert np.isnan(result.loc["BRA", "smoking"])


def test_fill_from_past_does_not_fill_a_leading_gap():
    df = make_df([("BRA", "AMR", 2000, np.nan, 1.0, 1.0), ("BRA", "AMR", 2010, 5.0, 1.0, 1.0)])
    assert np.isnan(fill_from_past(df, ["smoking"]).iloc[0]["smoking"])


def test_fill_from_past_orders_by_year_even_if_input_is_shuffled():
    df = make_df([
        ("BRA", "AMR", 2020, np.nan, 1.0, 1.0),
        ("BRA", "AMR", 2015, 20.0, 1.0, 1.0),
    ])
    result = fill_from_past(df, ["smoking"])
    assert result["year"].tolist() == [2015, 2020]
    assert result["smoking"].tolist() == [20.0, 20.0]


def test_build_modeling_frame_filters_years_and_fills_before_filtering():
    df = make_df([
        ("BRA", "AMR", 2000, 40.0, np.nan, 1.0),
        ("BRA", "AMR", 2010, np.nan, 2.0, 2.0),
    ])
    frame = build_modeling_frame(df, ["smoking", "pm25"], ["target"], 2010, ["smoking"])
    assert frame["year"].tolist() == [2010]
    assert frame.loc[0, "smoking"] == 40.0  # traz dados desde 2000, como filtrado


def test_build_modeling_frame_drops_rows_without_target_and_fills_region():
    df = make_df([
        ("BRA", None, 2015, 1.0, 1.0, 5.0),
        ("ARG", "AMR", 2015, 1.0, 1.0, np.nan),
    ])
    frame = build_modeling_frame(df, ["smoking", "pm25"], ["target"], 2010)
    assert frame["country_code"].tolist() == ["BRA"]
    assert frame.loc[0, "region"] == "UNKNOWN"


def test_build_modeling_frame_raises_on_missing_columns():
    df = make_df([("BRA", "AMR", 2015, 1.0, 1.0, 1.0)])
    with pytest.raises(KeyError, match="nope"):
        build_modeling_frame(df, ["smoking", "nope"], ["target"], 2010)


def test_build_modeling_frame_returns_only_requested_columns():
    df = make_df([("BRA", "AMR", 2015, 1.0, 1.0, 1.0)])
    frame = build_modeling_frame(df, ["pm25"], ["target"], 2010)
    assert list(frame.columns) == ["country_code", "region", "year", "pm25", "target"]