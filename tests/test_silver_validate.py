import pytest

from data_health_ftw.common.quality import DataQualityError
from data_health_ftw.silver.validate import validate_silver

SCHEMA = (
    "country_code string, region string, year int, "
    "indicator_code string, numeric_value double"
)


def make_silver(spark, rows):
    return spark.createDataFrame(rows, SCHEMA)


def valid_rows():
    return [
        ("BRA", "AMR", 2020, "A", 1.0),
        ("BRA", "AMR", 2020, "B", 2.0),
    ]


def test_validate_silver_passes_on_clean_data(spark):
    validate_silver(make_silver(spark, valid_rows()), expected_indicators=2)


def test_validate_silver_raises_on_empty(spark):
    with pytest.raises(DataQualityError, match="empty"):
        validate_silver(make_silver(spark, []), expected_indicators=2)


def test_validate_silver_raises_on_nan_value(spark):
    rows = valid_rows() + [("ARG", "AMR", 2020, "A", float("nan"))]
    with pytest.raises(DataQualityError, match="nulls"):
        validate_silver(make_silver(spark, rows), expected_indicators=2)


def test_validate_silver_raises_on_blank_string(spark):
    rows = valid_rows() + [("ARG", "  ", 2020, "A", 3.0)]
    with pytest.raises(DataQualityError, match="empty strings"):
        validate_silver(make_silver(spark, rows), expected_indicators=2)


def test_validate_silver_raises_on_duplicated_keys(spark):
    rows = valid_rows() + [("BRA", "AMR", 2020, "A", 9.0)]
    with pytest.raises(DataQualityError, match="duplicated"):
        validate_silver(make_silver(spark, rows), expected_indicators=2)


def test_validate_silver_raises_on_unexpected_indicator_count(spark):
    with pytest.raises(DataQualityError, match="distinct values"):
        validate_silver(make_silver(spark, valid_rows()), expected_indicators=15)


def test_validate_silver_raises_on_wrong_value_type(spark):
    base = make_silver(spark, valid_rows())
    df = base.withColumn("numeric_value", base["numeric_value"].cast("string"))
    with pytest.raises(DataQualityError, match="expected double"):
        validate_silver(df, expected_indicators=2)
