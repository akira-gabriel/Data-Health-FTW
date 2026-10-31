import logging

from pyspark.sql import SparkSession

from data_health_ftw.common import config
from data_health_ftw.common import quality as q
from data_health_ftw.common.delta_io import overwrite_table
from data_health_ftw.silver import ghe_transform as t
from data_health_ftw.silver.ghe_validate import validate_ghe_silver
from data_health_ftw.silver.transform_gho import select_latest_snapshot

logger = logging.getLogger(__name__)


def run(spark: SparkSession) -> None:
    bronze_df = spark.read.table(config.GHE_BRONZE_TABLE)
    latest_df = select_latest_snapshot(bronze_df)
    selected_df = t.select_target_rows(latest_df, config.GHE_CAUSES, config.GHE_SEX)
    silver_df = t.to_silver_schema(selected_df, config.GHE_CAUSES)

    validate_ghe_silver(silver_df)
    expected_rows = silver_df.count()

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.SILVER_SCHEMA}")
    overwrite_table(silver_df, config.GHE_SILVER_TABLE)

    q.assert_row_count(spark.table(config.GHE_SILVER_TABLE).count(), expected_rows, config.GHE_SILVER_TABLE)
    logger.info("GHE silver: %d rows", expected_rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run(SparkSession.builder.getOrCreate())


if __name__ == "__main__":
    main()
