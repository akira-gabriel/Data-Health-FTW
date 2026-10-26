import logging

from pyspark.sql import DataFrame, SparkSession

from data_health_ftw.common import config
from data_health_ftw.common import quality as q

logger = logging.getLogger(__name__)


def validate_silver(
    df: DataFrame,
    expected_indicators: int = config.EXPECTED_INDICATOR_COUNT,
    table_name: str = config.SILVER_TABLE,
) -> None:
    """Raise DataQualityError if the silver frame breaks its contract."""
    q.assert_not_empty(df, table_name)
    q.assert_no_nulls(df, config.SILVER_REQUIRED_COLUMNS, table_name)
    q.assert_no_empty_strings(df, config.SILVER_REQUIRED_COLUMNS, table_name)
    q.assert_unique_key(df, config.SILVER_UNIQUE_KEY, table_name)
    q.assert_distinct_count(df, "indicator_code", expected_indicators, table_name)
    q.assert_column_type(df, "numeric_value", "double", table_name)


def run(spark: SparkSession) -> None:
    validate_silver(spark.read.table(config.SILVER_TABLE))
    logger.info("Silver validation passed")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run(SparkSession.builder.getOrCreate())


if __name__ == "__main__":
    main()