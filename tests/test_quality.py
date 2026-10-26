import pytest

from data_health_ftw.common import quality as q


def test_count_missing_treats_nan_as_missing_for_floats(spark):
    df = spark.createDataFrame([(1.0,), (float("nan"),), (None,)], "value double")
    assert q.count_missing(df, ["value"]) == {"value": 2}


def test_count_missing_on_empty_frame_is_zero(spark):
    df = spark.createDataFrame([], "value double")
    assert q.count_missing(df, ["value"]) == {"value": 0}


def test_compute_null_rates(spark):
    df = spark.createDataFrame([(1.0,), (None,), (None,), (4.0,)], "value double")
    assert q.compute_null_rates(df, ["value"]) == {"value": 0.5}


def test_assert_not_empty_raises(spark):
    with pytest.raises(q.DataQualityError, match="empty"):
        q.assert_not_empty(spark.createDataFrame([], "x int"), "t")


def test_assert_no_nulls_raises(spark):
    df = spark.createDataFrame([("a",), (None,)], "x string")
    with pytest.raises(q.DataQualityError, match="nulls"):
        q.assert_no_nulls(df, ["x"], "t")


def test_assert_not_all_null_raises_when_column_is_entirely_null(spark):
    df = spark.createDataFrame([(None, 1), (None, 2)], "a string, b int")
    with pytest.raises(q.DataQualityError, match="entirely null"):
        q.assert_not_all_null(df, ["a", "b"], "t")


def test_assert_no_empty_strings_raises_on_blank_values(spark):
    df = spark.createDataFrame([("a",), ("  ",)], "x string")
    with pytest.raises(q.DataQualityError, match="empty strings"):
        q.assert_no_empty_strings(df, ["x"], "t")


def test_assert_no_empty_strings_ignores_non_string_columns(spark):
    df = spark.createDataFrame([(1,), (2,)], "x int")
    q.assert_no_empty_strings(df, ["x"], "t")


def test_assert_unique_key_raises_on_duplicates(spark):
    df = spark.createDataFrame([("BRA", 2020), ("BRA", 2020)], "c string, y int")
    with pytest.raises(q.DataQualityError, match="duplicated"):
        q.assert_unique_key(df, ["c", "y"], "t")


def test_assert_column_type(spark):
    df = spark.createDataFrame([(1.0,)], "v double")
    q.assert_column_type(df, "v", "double", "t")
    with pytest.raises(q.DataQualityError, match="expected int"):
        q.assert_column_type(df, "v", "int", "t")


def test_assert_distinct_count(spark):
    df = spark.createDataFrame([("a",), ("a",), ("b",)], "x string")
    q.assert_distinct_count(df, "x", 2, "t")
    with pytest.raises(q.DataQualityError):
        q.assert_distinct_count(df, "x", 3, "t")


def test_assert_row_count():
    q.assert_row_count(5, 5, "t")
    with pytest.raises(q.DataQualityError):
        q.assert_row_count(4, 5, "t")
