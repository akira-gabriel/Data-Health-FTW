import pytest
from pyspark.sql.types import StringType, StructField, StructType

from data_health_ftw.common.config import Indicator
from data_health_ftw.silver.transform_gho import (
    get_rejected_rows,
    get_valid_rows,
    select_country_indicators,
    select_latest_snapshot,
    to_silver_schema,
)

BRONZE_COLUMNS = [
    "SpatialDimType", "SpatialDim", "ParentLocationCode", "TimeDim",
    "IndicatorCode", "NumericValue", "Dim1", "Dim2", "Dim3",
]
BRONZE_SCHEMA = StructType([StructField(c, StringType(), True) for c in BRONZE_COLUMNS])

INDICATORS = (
    Indicator("HIV_0000000026", "hiv"),
    Indicator("BP_04", "bp", "SEX_BTSX", "AGEGROUP_YEARS30-79"),
)


def make_row(**overrides):
    base = dict(
        SpatialDimType="COUNTRY", SpatialDim="BRA", ParentLocationCode="AMR",
        TimeDim="2020", IndicatorCode="HIV_0000000026", NumericValue="1.5",
        Dim1=None, Dim2=None, Dim3=None,
    )
    base.update(overrides)
    return tuple(base[c] for c in BRONZE_COLUMNS)


def make_bronze(spark, rows):
    return spark.createDataFrame(rows, BRONZE_SCHEMA)


def test_select_country_indicators_applies_dimension_rules(spark):
    df = make_bronze(spark, [
        make_row(),                                                     # kept
        make_row(Dim1="SEX_BTSX"),                                      # HIV with extra dimension
        make_row(IndicatorCode="BP_04", Dim1="SEX_BTSX",
                 Dim2="AGEGROUP_YEARS30-79"),                           # kept
        make_row(IndicatorCode="BP_04", Dim1="SEX_BTSX",
                 Dim2="AGEGROUP_YEARS18-PLUS"),                         # wrong age group
        make_row(SpatialDimType="REGION"),                              # not a country
        make_row(IndicatorCode="UNKNOWN"),                              # no rule
    ])
    assert select_country_indicators(df, INDICATORS).count() == 2


def test_select_country_indicators_rejects_empty_indicators(spark):
    with pytest.raises(ValueError):
        select_country_indicators(make_bronze(spark, [make_row()]), ())


def test_valid_and_rejected_rows_partition_the_input(spark):
    df = make_bronze(spark, [make_row(), make_row(NumericValue=None)])
    assert get_valid_rows(df).count() == 1
    assert get_rejected_rows(df).count() == 1


def test_to_silver_schema_renames_and_casts(spark):
    result = to_silver_schema(make_bronze(spark, [make_row()]))
    assert result.columns == [
        "country_code", "region", "year", "indicator_code", "numeric_value"
    ]
    assert dict(result.dtypes) == {
        "country_code": "string", "region": "string", "year": "int",
        "indicator_code": "string", "numeric_value": "double",
    }


def test_select_latest_snapshot_keeps_only_newest_date(spark):
    df = spark.createDataFrame(
        [("BRA", "2026-10-23"), ("BRA", "2026-10-24"), ("ARG", "2026-10-24")],
        ["country", "_ingestion_date"],
    )
    result = select_latest_snapshot(df)
    assert result.count() == 2
    assert result.select("_ingestion_date").distinct().count() == 1
