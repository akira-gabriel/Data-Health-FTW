from pyspark.sql import DataFrame


def overwrite_table(df: DataFrame, table: str) -> None:
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table)
    )