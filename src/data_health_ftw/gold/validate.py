import logging
from collections.abc import Sequence

from pyspark.sql import DataFrame, SparkSession

from data_health_ftw.common import config
from data_health_ftw.common import quality as q

logger = logging.getLogger(__name__)


def validate_latest(df: DataFrame, table_name: str = config.GOLD_LATEST_TABLE) -> None:
    q.assert_not_empty(df, table_name)
    q.assert_no_nulls(df, config.GOLD_LATEST_REQUIRED_COLUMNS, table_name)
    q.assert_unique_key(df, config.GOLD_LATEST_UNIQUE_KEY, table_name)


def validate_wide(
    df: DataFrame,
    indicator_names: Sequence[str] = tuple(config.INDICATOR_NAMES),
    table_name: str = config.GOLD_WIDE_TABLE,
) -> None:
    q.assert_not_empty(df, table_name)
    q.assert_no_nulls(df, config.GOLD_WIDE_REQUIRED_COLUMNS, table_name)
    q.assert_not_all_null(df, list(indicator_names), table_name)
    q.assert_unique_key(df, config.GOLD_WIDE_UNIQUE_KEY, table_name)

    expected_columns = set(config.GOLD_WIDE_ID_COLUMNS) | set(indicator_names)
    if set(df.columns) != expected_columns:
        raise q.DataQualityError(
            f"{table_name}: unexpected columns. "
            f"Missing: {sorted(expected_columns - set(df.columns))}, "
            f"extra: {sorted(set(df.columns) - expected_columns)}"
        )


def run(spark: SparkSession) -> None:
    latest_df = spark.read.table(config.GOLD_LATEST_TABLE)
    wide_df = spark.read.table(config.GOLD_WIDE_TABLE)

    validate_latest(latest_df)
    validate_wide(wide_df)

    rates = q.compute_null_rates(wide_df, config.INDICATOR_NAMES)
    for name, rate in sorted(rates.items(), key=lambda item: item[1], reverse=True):
        logger.info("null rate %-35s %5.1f%%", name, rate * 100)
    logger.info("Gold validation passed")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run(SparkSession.builder.getOrCreate())


if __name__ == "__main__":
    main()