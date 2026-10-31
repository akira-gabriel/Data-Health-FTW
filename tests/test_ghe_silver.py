import pytest

from data_health_ftw.common.config import GheCause
from data_health_ftw.common.quality import DataQualityError
from data_health_ftw.silver.ghe_transform import select_target_rows, to_silver_schema
from data_health_ftw.silver.ghe_validate import validate_ghe_silver

CAUSES = (GheCause(1100, "cvd_asdr"), GheCause(1180, "copd_asdr"))

BRONZE_SCHEMA = (
    "country_code string, year string, sex string, ghe_code string, "
    "cause string, asdr_per_100k string"
)
SILVER_SCHEMA = (
    "country_code string, year int, ghe_code int, cause_name string, asdr_per_100k double"
)


def make_bronze(spark, rows):
    return spark.createDataFrame(rows, BRONZE_SCHEMA)


def make_silver(spark, rows):
    return spark.createDataFrame(rows, SILVER_SCHEMA)


def test_select_target_rows_filters_sex_and_causes(spark):
    df = make_bronze(spark, [
        ("BRA", "2023", "Persons", "1100", "CVD", "300.0"),
        ("BRA", "2023", "Males", "1100", "CVD", "400.0"),
        ("BRA", "2023", "Persons", "9999", "Other", "1.0"),
    ])
    result = select_target_rows(df, CAUSES, "Persons")
    assert [(r["sex"], r["ghe_code"]) for r in result.collect()] == [("Persons", "1100")]


def test_select_target_rows_rejects_empty_causes(spark):
    with pytest.raises(ValueError):
        select_target_rows(make_bronze(spark, []), (), "Persons")


def test_to_silver_schema_casts_types_and_maps_names(spark):
    df = make_bronze(spark, [("BRA", "2023", "Persons", "1180", "COPD", "49.5")])
    result = to_silver_schema(df, CAUSES)
    row = result.first()
    assert (row["year"], row["ghe_code"], row["cause_name"], row["asdr_per_100k"]) == (
        2023, 1180, "copd_asdr", 49.5
    )
    assert dict(result.dtypes) == {
        "country_code": "string", "year": "int", "ghe_code": "int",
        "cause_name": "string", "asdr_per_100k": "double",
    }


def valid_silver(spark):
    return make_silver(spark, [
        ("BRA", 2023, 1100, "cvd_asdr", 300.0),
        ("BRA", 2023, 1180, "copd_asdr", 50.0),
        ("BRA", 2022, 1100, "cvd_asdr", 310.0),
        ("BRA", 2022, 1180, "copd_asdr", 55.0),
    ])


def test_validate_passes_on_valid_frame(spark):
    validate_ghe_silver(valid_silver(spark), expected_causes=2, expected_years=2)


def test_validate_rejects_duplicate_keys(spark):
    df = valid_silver(spark).union(make_silver(spark, [("BRA", 2023, 1100, "cvd_asdr", 1.0)]))
    with pytest.raises(DataQualityError, match="duplicated"):
        validate_ghe_silver(df, expected_causes=2, expected_years=2)


def test_validate_rejects_null_rates(spark):
    df = valid_silver(spark).union(make_silver(spark, [("ARG", 2023, 1100, "cvd_asdr", None)]))
    with pytest.raises(DataQualityError, match="nulls"):
        validate_ghe_silver(df, expected_causes=2, expected_years=2)


def test_validate_rejects_negative_rates(spark):
    df = valid_silver(spark).union(make_silver(spark, [("ARG", 2023, 1100, "cvd_asdr", -1.0)]))
    with pytest.raises(DataQualityError, match="negative"):
        validate_ghe_silver(df, expected_causes=2, expected_years=2)


def test_validate_rejects_missing_year(spark):
    with pytest.raises(DataQualityError, match="distinct values in year"):
        validate_ghe_silver(valid_silver(spark), expected_causes=2, expected_years=8)


def test_validate_rejects_empty_frame(spark):
    with pytest.raises(DataQualityError, match="empty"):
        validate_ghe_silver(make_silver(spark, []), expected_causes=2, expected_years=2)