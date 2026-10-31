from pyspark.sql import DataFrame, SparkSession


def overwrite_snapshot(
    spark: SparkSession, df: DataFrame, table: str, date_column: str, date_value: str
) -> None:
    """Idempotently write one dated snapshot: re-running a day replaces only that day.

    replaceWhere needs an existing table, so the first run does a plain overwrite.
    """
    writer = df.write.format("delta").mode("overwrite").option("mergeSchema", "true")
    if spark.catalog.tableExists(table):
        writer = writer.option("replaceWhere", f"{date_column} = '{date_value}'")
    writer.saveAsTable(table)


def overwrite_table(df: DataFrame, table: str) -> None:
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table)
    )