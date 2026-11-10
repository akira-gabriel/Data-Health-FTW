import logging
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from data_health_ftw.bronze import gho_client, transform
from data_health_ftw.common import config

logger = logging.getLogger(__name__)


def run(spark: SparkSession) -> None:
    payloads, failed = gho_client.fetch_all_payloads(
        indicators=config.INDICATORS,
        base_url=config.GHO_BASE_URL,
        timeout=config.REQUEST_TIMEOUT_SECONDS,
        pause_seconds=config.PAUSE_BETWEEN_REQUESTS_SECONDS,
        page_size=config.GHO_PAGE_SIZE,
    )
    if failed:
        raise RuntimeError(f"Ingestion failed for indicators: {failed}")
    countries = gho_client.fetch_country_reference(
        config.REFMART_BASE_URL, config.REQUEST_TIMEOUT_SECONDS, config.GHO_PAGE_SIZE
    )

    ingested_at = datetime.now(timezone.utc)
    ingestion_date = ingested_at.date().isoformat()

    raw_df = transform.build_raw_frame(payloads, config.GHO_COLUMN_MAP)
    raw_df = transform.attach_country_reference(raw_df, countries)
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


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run(SparkSession.builder.getOrCreate())


if __name__ == "__main__":
    main()