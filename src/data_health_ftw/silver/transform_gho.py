from collections.abc import Sequence
from functools import reduce
from operator import or_

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from data_health_ftw.common.config import Indicator


def select_latest_snapshot(df: DataFrame, date_column: str = "_ingestion_date") -> DataFrame:
    latest_date = df.agg(F.max(date_column)).first()[0]
    return df.filter(F.col(date_column) == latest_date)


def _dimension_matches(column_name: str, expected: str | None) -> Column:
    column = F.col(column_name)
    return column.isNull() if expected is None else column == expected


def build_indicator_condition(indicator: Indicator) -> Column:
    return (
        (F.col("IndicatorCode") == indicator.code)
        & _dimension_matches("Dim1", indicator.dim1)
        & _dimension_matches("Dim2", indicator.dim2)
        & _dimension_matches("Dim3", indicator.dim3)
    )


def select_country_indicators(df: DataFrame, indicators: Sequence[Indicator]) -> DataFrame:
    """Keep country-level rows that match each indicator's dimension rule."""
    if not indicators:
        raise ValueError("indicators must not be empty")
    condition = reduce(or_, (build_indicator_condition(i) for i in indicators))
    return df.filter(F.col("SpatialDimType") == "COUNTRY").filter(condition)


def get_valid_rows(df: DataFrame) -> DataFrame:
    return df.filter(F.col("NumericValue").isNotNull())


def get_rejected_rows(df: DataFrame) -> DataFrame:
    return df.filter(F.col("NumericValue").isNull())


def to_silver_schema(df: DataFrame) -> DataFrame:
    return df.select(
        F.col("SpatialDim").alias("country_code"),
        F.col("ParentLocationCode").alias("region"),
        F.col("TimeDim").cast("int").alias("year"),
        F.col("IndicatorCode").alias("indicator_code"),
        F.col("NumericValue").cast("double").alias("numeric_value"),
    )