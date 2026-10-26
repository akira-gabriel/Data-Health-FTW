import pytest

from data_health_ftw.common.config import Indicator
from data_health_ftw.common.quality import DataQualityError
from data_health_ftw.gold.transform import (
    add_indicator_names,
    build_latest,
    build_wide,
    find_unmapped_codes,
)
from data_health_ftw.gold.validate import validate_latest, validate_wide

INDICATORS = (Indicator("A", "alpha"), Indicator("B", "beta"))
NAMES = [i.name for i in INDICATORS]

SILVER_SCHEMA = (
    "country_code string, region string, year int, "
    "indicator_code string, numeric_value double"
)


def make_silver(spark, rows):
    return spark.createDataFrame(rows, SILVER_SCHEMA)


def test_add_indicator_names_maps_codes(spark):
    df = make_silver(spark, [("BRA", "AMR", 2020, "A", 1.0)])
    result = add_indicator_names(df, INDICATORS)
    assert result.first()["indicator"] == "alpha"


def test_find_unmapped_codes_lists_unknown_codes(spark):
    df = make_silver(spark, [("BRA", "AMR", 2020, "A", 1.0), ("BRA", "AMR", 2020, "Z", 2.0)])
    named = add_indicator_names(df, INDICATORS)
    assert find_unmapped_codes(named) == ["Z"]


def test_build_latest_keeps_most_recent_year(spark):
    df = make_silver(spark, [
        ("BRA", "AMR", 2019, "A", 1.0),
        ("BRA", "AMR", 2020, "A", 2.0),
        ("BRA", "AMR", 2018, "B", 5.0),
    ])
    result = build_latest(add_indicator_names(df, INDICATORS))
    rows = {r["indicator"]: (r["year"], r["numeric_value"]) for r in result.collect()}
    assert rows == {"alpha": (2020, 2.0), "beta": (2018, 5.0)}
    assert "indicator_code" not in result.columns


def test_build_wide_pivots_indicators_into_columns(spark):
    df = make_silver(spark, [
        ("BRA", "AMR", 2020, "A", 1.0),
        ("BRA", "AMR", 2020, "B", 2.0),
        ("ARG", "AMR", 2020, "A", 3.0),
    ])
    result = build_wide(add_indicator_names(df, INDICATORS), NAMES)
    assert set(result.columns) == {"country_code", "region", "year", "alpha", "beta"}
    by_country = {r["country_code"]: r for r in result.collect()}
    assert by_country["BRA"]["alpha"] == 1.0
    assert by_country["BRA"]["beta"] == 2.0
    assert by_country["ARG"]["beta"] is None


def test_validate_latest_raises_on_duplicated_keys(spark):
    df = spark.createDataFrame(
        [("BRA", "alpha", 2020, 1.0), ("BRA", "alpha", 2019, 2.0)],
        "country_code string, indicator string, year int, numeric_value double",
    )
    with pytest.raises(DataQualityError, match="duplicated"):
        validate_latest(df)


def test_validate_wide_raises_on_unexpected_columns(spark):
    df = spark.createDataFrame(
        [("BRA", "AMR", 2020, 1.0)],
        "country_code string, region string, year int, alpha double",
    )
    with pytest.raises(DataQualityError, match="unexpected columns"):
        validate_wide(df, NAMES)


def test_validate_wide_raises_when_an_indicator_is_entirely_null(spark):
    df = spark.createDataFrame(
        [("BRA", "AMR", 2020, 1.0, None), ("ARG", "AMR", 2020, 2.0, None)],
        "country_code string, region string, year int, alpha double, beta double",
    )
    with pytest.raises(DataQualityError, match="entirely null"):
        validate_wide(df, NAMES)
