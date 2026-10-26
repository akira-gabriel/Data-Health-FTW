import logging
from datetime import datetime, timezone
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.bronze import gho_client, transform
from src.common import config

logger = logging.getLogger(__name__)

def run(spark: SparkSession) -> None:
    payloads, failed = gho_client.fetch_all_payloads(
        codes=config.INDICATOR_CODES,
        base_url=config.GHO_BASE_URL,
        timeout=config.REQUEST_TIMEOUT_SECONDS,
        pause_seconds=config.PAUSE_BETWEEN_REQUESTS_SECONDS,
    )
    if failed:
        raise RuntimeError(f"Ingestion failed for indicators: {failed}")

    ingested_at = datetime.now(timezone.utc)
    ingestion_date = ingested_at.date().isoformat()

    raw_df = transform.build_raw_frame(payloads)
    raw_df = transform.add_ingestion_metadata(raw_df, config.GHO_BASE_URL, ingested_at)
    raw_df = transform.cast_to_string(raw_df)

    bronze_df = (
        spark.createDataFrame(raw_df)
        .withColumn("_ingested_at", F.to_timestamp("_ingested_at"))
        .withColumn("_ingestion_date", F.to_date("_ingestion_date"))
    )

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.BRONZE_SCHEMA}")
    (
        bronze_df.write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", f"_ingestion_date = '{ingestion_date}'")
        .option("mergeSchema", "true")
        .saveAsTable(config.BRONZE_TABLE)
    )
    logger.info("Snapshot %s written: %d rows", ingestion_date, bronze_df.count())