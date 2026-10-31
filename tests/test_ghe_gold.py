import pytest

from data_health_ftw.common import config
from data_health_ftw.common.quality import DataQualityError
from data_health_ftw.gold.ml_transform import build_ml_features, build_targets
from data_health_ftw.gold.ml_validate import validate_ml_features

GHE_SCHEMA = "country_code string, year int, ghe_code int, cause_name string, asdr_per_100k double"
NAMES = ["copd_asdr", "asthma_asdr", "cvd_asdr"]
COMPONENTS = ("copd_asdr", "asthma_asdr")


def make_ghe(spark, rows):
    return spark.createDataFrame(rows, GHE_SCHEMA)


def ghe_rows(country, year, copd, asthma, cvd):
    return [
        (country, year, 1180, "copd_asdr", copd),
        (country, year, 1190, "asthma_asdr", asthma),
        (country, year, 1100, "cvd_asdr", cvd),
    ]


def test_build_targets_pivots_causes_and_sums_chronic_target(spark):
    df = make_ghe(spark, ghe_rows("BRA", 2023, 49.5, 2.1, 300.0))
    result = build_targets(df, NAMES, "chronic_resp_asdr", COMPONENTS).first()
    assert result["copd_asdr"] == 49.5
    assert result["cvd_asdr"] == 300.0
    assert result["chronic_resp_asdr"] == pytest.approx(51.6)


def test_build_targets_gives_one_row_per_country_year(spark):
    df = make_ghe(spark, ghe_rows("BRA", 2023, 1.0, 1.0, 1.0) + ghe_rows("BRA", 2022, 2.0, 2.0, 2.0))
    assert build_targets(df, NAMES, "chronic_resp_asdr", COMPONENTS).count() == 2


def test_build_ml_features_left_join_keeps_every_ghe_row(spark):
    targets = build_targets(
        make_ghe(spark, ghe_rows("BRA", 2023, 1.0, 1.0, 1.0) + ghe_rows("ARG", 2023, 2.0, 2.0, 2.0)),
        NAMES, "chronic_resp_asdr", COMPONENTS,
    )
    wide = spark.createDataFrame(
        [("BRA", "AMR", 2023, 12.0)], "country_code string, region string, year int, pm25_exposure double"
    )
    result = build_ml_features(targets, wide, config.GOLD_ML_ID_COLUMNS)
    rows = {r["country_code"]: r for r in result.collect()}
    assert result.count() == 2
    assert rows["BRA"]["pm25_exposure"] == 12.0
    assert rows["ARG"]["pm25_exposure"] is None  # no GHO indicators for this country-year


def test_build_ml_features_puts_id_columns_first(spark):
    targets = build_targets(
        make_ghe(spark, ghe_rows("BRA", 2023, 1.0, 1.0, 1.0)), NAMES, "chronic_resp_asdr", COMPONENTS
    )
    wide = spark.createDataFrame(
        [("BRA", "AMR", 2023, 12.0)], "country_code string, region string, year int, pm25_exposure double"
    )
    result = build_ml_features(targets, wide, config.GOLD_ML_ID_COLUMNS)
    assert result.columns[:3] == ["country_code", "region", "year"]


# --- validate_ml_features -------------------------------------------------------

INDICATORS = ["pm25_exposure", "smoking_prev"]


def make_features(spark, rows=None):
    columns = config.GOLD_ML_ID_COLUMNS + config.GOLD_ML_TARGET_COLUMNS + INDICATORS
    n_targets = len(config.GOLD_ML_TARGET_COLUMNS)
    default = [("BRA", "AMR", 2023, *([1.0] * n_targets), 12.0, 20.0)]
    schema = "country_code string, region string, year int, " + ", ".join(
        f"{c} double" for c in config.GOLD_ML_TARGET_COLUMNS + INDICATORS
    )
    return spark.createDataFrame(rows if rows is not None else default, schema), columns


def test_validate_ml_features_passes_on_valid_frame(spark):
    df, _ = make_features(spark)
    validate_ml_features(df, indicator_names=INDICATORS)


def test_validate_ml_features_rejects_missing_column(spark):
    df, _ = make_features(spark)
    with pytest.raises(DataQualityError, match="unexpected columns"):
        validate_ml_features(df.drop("smoking_prev"), indicator_names=INDICATORS)


def test_validate_ml_features_rejects_null_target(spark):
    n_targets = len(config.GOLD_ML_TARGET_COLUMNS)
    df, _ = make_features(spark, [("BRA", "AMR", 2023, None, *([1.0] * (n_targets - 1)), 12.0, 20.0)])
    with pytest.raises(DataQualityError, match="nulls"):
        validate_ml_features(df, indicator_names=INDICATORS)


def test_validate_ml_features_rejects_duplicate_keys(spark):
    n_targets = len(config.GOLD_ML_TARGET_COLUMNS)
    row = ("BRA", "AMR", 2023, *([1.0] * n_targets), 12.0, 20.0)
    df, _ = make_features(spark, [row, row])
    with pytest.raises(DataQualityError, match="duplicated"):
        validate_ml_features(df, indicator_names=INDICATORS)


def test_validate_ml_features_rejects_empty_frame(spark):
    df, _ = make_features(spark, [])
    with pytest.raises(DataQualityError, match="empty"):
        validate_ml_features(df, indicator_names=INDICATORS)