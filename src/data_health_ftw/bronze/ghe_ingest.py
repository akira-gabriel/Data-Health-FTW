import logging
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from data_health_ftw.bronze import transform
from data_health_ftw.common import config
from data_health_ftw.common.delta_io import overwrite_snapshot
from data_health_ftw.ghe.parser import parse_asdr_workbook

logger = logging.getLogger(__name__)


def run(spark: SparkSession, path: str = config.GHE_ASDR_PATH) -> None:
    ingested_at = datetime.now(timezone.utc)
    ingestion_date = ingested_at.date().isoformat()

    raw_df = parse_asdr_workbook(path)
    raw_df = transform.add_ingestion_metadata(raw_df, path, ingested_at)
    raw_df = transform.cast_to_string(raw_df)

    bronze_df = (
        spark.createDataFrame(raw_df)
        .withColumn("_ingested_at", F.to_timestamp("_ingested_at"))
        .withColumn("_ingestion_date", F.to_date("_ingestion_date"))
    )

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.BRONZE_SCHEMA}")
    overwrite_snapshot(spark, bronze_df, config.GHE_BRONZE_TABLE, "_ingestion_date", ingestion_date)
    logger.info("GHE snapshot %s written: %d rows from %s", ingestion_date, bronze_df.count(), path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run(SparkSession.builder.getOrCreate())


if __name__ == "__main__":
    main()
