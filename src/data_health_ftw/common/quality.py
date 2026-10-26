from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DataType, DoubleType, FloatType, StringType

FLOAT_TYPES = (DoubleType, FloatType)


class DataQualityError(ValueError):

def _is_missing(column: str, dtype: DataType) -> Column:
    col = F.col(column)
    if isinstance(dtype, FLOAT_TYPES):
        return col.isNull() | F.isnan(col)
    return col.isNull()


def count_missing(df: DataFrame, columns: list[str]) -> dict[str, int]:
    schema = df.schema
    row = df.agg(
        *[F.sum(_is_missing(c, schema[c].dataType).cast("int")).alias(c) for c in columns]
    ).first()
    return {c: int(row[c] or 0) for c in columns}


def compute_null_rates(df: DataFrame, columns: list[str]) -> dict[str, float]:
    total = df.count()
    missing = count_missing(df, columns)
    return {c: (n / total if total else 0.0) for c, n in missing.items()}


def assert_not_empty(df: DataFrame, table_name: str) -> None:
    if df.isEmpty():
        raise DataQualityError(f"{table_name}: table is empty")


def assert_no_nulls(df: DataFrame, columns: list[str], table_name: str) -> None:
    failures = {c: n for c, n in count_missing(df, columns).items() if n > 0}
    if failures:
        raise DataQualityError(f"{table_name}: nulls in required columns: {failures}")


def assert_not_all_null(df: DataFrame, columns: list[str], table_name: str) -> None:
    total = df.count()
    if total == 0:
        raise DataQualityError(f"{table_name}: table is empty")
    all_null = [c for c, n in count_missing(df, columns).items() if n == total]
    if all_null:
        raise DataQualityError(f"{table_name}: columns entirely null: {all_null}")


def assert_no_empty_strings(df: DataFrame, columns: list[str], table_name: str) -> None:
    string_columns = [c for c in columns if isinstance(df.schema[c].dataType, StringType)]
    if not string_columns:
        return
    row = df.agg(
        *[F.sum((F.trim(F.col(c)) == "").cast("int")).alias(c) for c in string_columns]
    ).first()
    empty = {c: int(row[c]) for c in string_columns if row[c]}
    if empty:
        raise DataQualityError(f"{table_name}: empty strings in required columns: {empty}")


def assert_unique_key(df: DataFrame, key: list[str], table_name: str) -> None:
    duplicated = df.groupBy(*key).count().filter(F.col("count") > 1).count()
    if duplicated:
        raise DataQualityError(f"{table_name}: {duplicated} duplicated keys on {key}")


def assert_column_type(df: DataFrame, column: str, expected: str, table_name: str) -> None:
    actual = dict(df.dtypes)[column]
    if actual != expected:
        raise DataQualityError(f"{table_name}: {column} is {actual}, expected {expected}")


def assert_distinct_count(df: DataFrame, column: str, expected: int, table_name: str) -> None:
    actual = df.select(column).distinct().count()
    if actual != expected:
        raise DataQualityError(
            f"{table_name}: {actual} distinct values in {column}, expected {expected}"
        )


def assert_row_count(actual: int, expected: int, label: str) -> None:
    if actual != expected:
        raise DataQualityError(f"{label}: {actual} rows, expected {expected}")