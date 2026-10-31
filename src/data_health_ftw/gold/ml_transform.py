from collections.abc import Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_targets(
    ghe_df: DataFrame,
    cause_names: Sequence[str],
    chronic_target: str,
    chronic_components: Sequence[str],
) -> DataFrame:
    """One row per (country, year), one column per cause, plus the derived chronic target."""
    wide = (
        ghe_df.groupBy("country_code", "year")
        .pivot("cause_name", list(cause_names))  # explicit list avoids an extra Spark job
        .agg(F.first("asdr_per_100k"))
    )
    total = sum((F.col(c) for c in chronic_components[1:]), F.col(chronic_components[0]))
    return wide.withColumn(chronic_target, total)


def build_ml_features(
    targets_df: DataFrame, wide_df: DataFrame, id_columns: Sequence[str]
) -> DataFrame:
    """Attach the GHO indicators to every GHE (country, year).

    Left join: GHE defines the rows (it has the targets); indicators missing for a
    country-year stay null and are handled at modelling time.
    """
    joined = targets_df.join(wide_df, on=["country_code", "year"], how="left")
    ordered = list(id_columns) + [c for c in joined.columns if c not in id_columns]
    return joined.select(*ordered)
