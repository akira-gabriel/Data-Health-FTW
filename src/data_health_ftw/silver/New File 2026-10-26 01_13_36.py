import logging

from pyspark.sql import SparkSession

from data_health_ftw.common import config
from data_health_ftw.common import quality as q
from data_health_ftw.common.delta_io import overwrite_table
from data_health_ftw.silver import transform_gho as t
from data_health_ftw.silver.validate import validate_silver

logger = logging.getLogger(__name__)


def run(spark: SparkSession) -> None:
    bronze_df = spark.read.table(config.BRONZE_TABLE)
    latest_df = t.select_latest_snapshot(bronze_df)
    selected_df = t.select_country_indicators(latest_df, config.INDICATORS)

    silver_df = t.to_silver_schema(t.get_valid_rows(selected_df))
    rejected_df = t.get_rejected_rows(selected_df)

    # Fail before overwriting: a bad run must not replace a good silver table
    validate_silver(silver_df)

    expected_rows = selected_df.count()

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.SILVER_SCHEMA}")
    overwrite_table(silver_df, config.SILVER_TABLE)
    overwrite_table(rejected_df, config.SILVER_REJECTED_TABLE)

    silver_rows = spark.table(config.SILVER_TABLE).count()
    rejected_rows = spark.table(config.SILVER_REJECTED_TABLE).count()
    q.assert_row_count(silver_rows + rejected_rows, expected_rows, "silver + rejected")
    logger.info("Silver: %d rows, rejected: %d rows", silver_rows, rejected_rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run(SparkSession.builder.getOrCreate())


if __name__ == "__main__":
    main()