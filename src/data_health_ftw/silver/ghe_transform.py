from collections.abc import Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from data_health_ftw.common.config import GheCause


def select_target_rows(df: DataFrame, causes: Sequence[GheCause], sex: str) -> DataFrame:
    """Keep one sex block and only the causes the project uses (bronze stores codes as strings)."""
    if not causes:
        raise ValueError("causes must not be empty")
    codes = [str(c.code) for c in causes]
    return df.filter(F.col("sex") == sex).filter(F.col("ghe_code").isin(codes))


def to_silver_schema(df: DataFrame, causes: Sequence[GheCause]) -> DataFrame:
    mapping = F.create_map(*[F.lit(v) for c in causes for v in (str(c.code), c.name)])
    return df.select(
        F.col("country_code"),
        F.col("year").cast("int").alias("year"),
        F.col("ghe_code").cast("int").alias("ghe_code"),
        mapping[F.col("ghe_code")].alias("cause_name"),
        F.col("asdr_per_100k").cast("double").alias("asdr_per_100k"),
    )
