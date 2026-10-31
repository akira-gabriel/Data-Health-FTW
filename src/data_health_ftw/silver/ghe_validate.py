from pyspark.sql import DataFrame

from data_health_ftw.common import config
from data_health_ftw.common import quality as q


def validate_ghe_silver(
    df: DataFrame,
    expected_causes: int = len(config.GHE_CAUSES),
    expected_years: int = config.GHE_EXPECTED_YEARS,
    table_name: str = config.GHE_SILVER_TABLE,
) -> None:
    """Raise DataQualityError if the GHE silver frame breaks its contract."""
    q.assert_not_empty(df, table_name)
    q.assert_no_nulls(df, config.GHE_SILVER_REQUIRED_COLUMNS, table_name)
    q.assert_no_empty_strings(df, config.GHE_SILVER_REQUIRED_COLUMNS, table_name)
    q.assert_unique_key(df, config.GHE_SILVER_UNIQUE_KEY, table_name)
    q.assert_distinct_count(df, "cause_name", expected_causes, table_name)
    q.assert_distinct_count(df, "year", expected_years, table_name)
    q.assert_column_type(df, "asdr_per_100k", "double", table_name)
    q.assert_non_negative(df, ["asdr_per_100k"], table_name)
