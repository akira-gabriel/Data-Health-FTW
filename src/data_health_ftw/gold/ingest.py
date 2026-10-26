import logging
from pyspark.sql import SparkSession
from data_health_ftw.common import config
from data_health_ftw.common import quality as q
from data_health_ftw.common.delta_io import overwrite_table
from data_health_ftw.gold import transform as t
from data_health_ftw.gold.validate import validate_latest, validate_wide

logger = logging.getLogger(__name__)


def run(spark: SparkSession) -> None:
    silver_df = spark.read.table(config.SILVER_TABLE)
    named_df = t.add_indicator_names(silver_df, config.INDICATORS)

    unmapped = t.find_unmapped_codes(named_df)
    if unmapped:
        raise q.DataQualityError(f"indicator_code values missing from INDICATORS: {unmapped}")

    latest_df = t.build_latest(named_df)
    wide_df = t.build_wide(named_df, config.INDICATOR_NAMES)

    validate_latest(latest_df)
    validate_wide(wide_df)

    expected_latest = latest_df.count()
    expected_wide = wide_df.count()

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.GOLD_SCHEMA}")
    overwrite_table(latest_df, config.GOLD_LATEST_TABLE)
    overwrite_table(wide_df, config.GOLD_WIDE_TABLE)

    persisted_latest = spark.table(config.GOLD_LATEST_TABLE)
    persisted_wide = spark.table(config.GOLD_WIDE_TABLE)
    q.assert_row_count(persisted_latest.count(), expected_latest, config.GOLD_LATEST_TABLE)
    q.assert_row_count(persisted_wide.count(), expected_wide, config.GOLD_WIDE_TABLE)
    if set(persisted_wide.columns) != set(wide_df.columns):
        raise q.DataQualityError(f"{config.GOLD_WIDE_TABLE}: schema differs from DataFrame")

    logger.info("Gold: %d latest rows, %d wide rows", expected_latest, expected_wide)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run(SparkSession.builder.getOrCreate())


if __name__ == "__main__":
    main()