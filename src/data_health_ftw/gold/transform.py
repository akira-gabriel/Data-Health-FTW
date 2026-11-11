from collections.abc import Sequence
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from data_health_ftw.common import config
from data_health_ftw.common.config import Indicator


def add_indicator_names(df: DataFrame, indicators: Sequence[Indicator]) -> DataFrame:
    mapping = F.create_map(*[F.lit(v) for i in indicators for v in (i.code, i.name)])
    return df.withColumn("indicator", mapping[F.col("indicator_code")])


def find_unmapped_codes(named_df: DataFrame) -> list[str]:
    rows = (
        named_df.filter(F.col("indicator").isNull())
        .select("indicator_code")
        .distinct()
        .collect()
    )
    return sorted(row["indicator_code"] for row in rows)


def build_latest(named_df: DataFrame) -> DataFrame:
    window = Window.partitionBy("country_code", "indicator").orderBy(F.desc("year"))
    return (
        named_df.withColumn("rn", F.row_number().over(window))
        .filter("rn = 1")
        .drop("rn", "indicator_code")
    )


def build_wide(named_df: DataFrame, indicator_names: Sequence[str]) -> DataFrame:
    return (
        named_df.groupBy("country_code", "region", "year")
        .pivot("indicator", list(indicator_names))
        .agg(F.first("numeric_value"))
    )


def add_gov_health_spend(wide_df: DataFrame) -> DataFrame:
    share = F.when(
        F.col("che_ncu") > 0, F.col("gghed_ncu") / F.col("che_ncu")
    )
    return wide_df.withColumn(
        config.GOV_HEALTH_SPEND, F.col("che_per_capita_usd") * share
    )