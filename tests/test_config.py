from data_health_ftw.common import config


def test_indicator_codes_are_unique():
    assert len(set(config.INDICATOR_CODES)) == len(config.INDICATOR_CODES)


def test_indicator_names_are_unique():
    assert len(set(config.INDICATOR_NAMES)) == len(config.INDICATOR_NAMES)


def test_expected_indicator_count_matches_registry():
    assert config.EXPECTED_INDICATOR_COUNT == len(config.INDICATORS) == 13


def test_dimensions_are_filled_from_the_left():
    for indicator in config.INDICATORS:
        if indicator.dim1 is None:
            assert indicator.dim2 is None and indicator.dim3 is None
        if indicator.dim2 is None:
            assert indicator.dim3 is None


def test_every_indicator_has_an_xmart_entity():
    assert all(i.entity.startswith("RELAY_") for i in config.INDICATORS)


def test_column_map_targets_legacy_names_used_by_silver():
    silver_needs = {"IndicatorCode", "SpatialDimType", "TimeDim", "NumericValue", "Dim1", "Dim2", "Dim3"}
    assert silver_needs <= set(config.GHO_COLUMN_MAP.values())


def test_every_indicator_accepts_at_least_one_publish_state():
    assert all(len(i.publish_states) > 0 for i in config.INDICATORS)


def test_publish_state_is_in_the_column_map():
    assert config.GHO_COLUMN_MAP["DIM_PUBLISH_STATE_CODE"] == "PublishState"