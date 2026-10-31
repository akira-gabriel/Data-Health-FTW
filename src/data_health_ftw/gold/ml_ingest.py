import logging

from pyspark.sql import SparkSession

from data_health_ftw.common import config
from data_health_ftw.common import quality as q
from data_health_ftw.common.delta_io import overwrite_table
from data_health_ftw.gold import ml_transform as t
from data_health_ftw.gold.ml_validate import validate_ml_features

logger = logging.getLogger(__name__)


def run(spark: SparkSession) -> None:
    ghe_df = spark.read.table(config.GHE_SILVER_TABLE)
    wide_df = spark.read.table(config.GOLD_WIDE_TABLE)

    targets_df = t.build_targets(
        ghe_df,
        config.GHE_CAUSE_NAMES,
        config.CHRONIC_RESP_TARGET,
        config.CHRONIC_RESP_COMPONENTS,
    )
    features_df = t.build_ml_features(targets_df, wide_df, config.GOLD_ML_ID_COLUMNS)

    validate_ml_features(features_df)
    expected_rows = targets_df.count()
    q.assert_row_count(features_df.count(), expected_rows, "ml_features vs GHE targets (join fan-out)")

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {config.GOLD_SCHEMA}")
    overwrite_table(features_df, config.GOLD_ML_FEATURES_TABLE)

    persisted = spark.table(config.GOLD_ML_FEATURES_TABLE)
    q.assert_row_count(persisted.count(), expected_rows, config.GOLD_ML_FEATURES_TABLE)
    logger.info("ML features: %d rows", expected_rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run(SparkSession.builder.getOrCreate())


if __name__ == "__main__":
    main()
