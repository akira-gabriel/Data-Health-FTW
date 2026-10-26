from data_health_ftw.common import config


def test_indicator_codes_are_unique():
    assert len(set(config.INDICATOR_CODES)) == len(config.INDICATOR_CODES)


def test_indicator_names_are_unique():
    assert len(set(config.INDICATOR_NAMES)) == len(config.INDICATOR_NAMES)


def test_expected_indicator_count_matches_registry():
    assert config.EXPECTED_INDICATOR_COUNT == len(config.INDICATORS) == 15


def test_dimensions_are_filled_from_the_left():
    # a later dimension never has a value while an earlier one is None
    for indicator in config.INDICATORS:
        if indicator.dim1 is None:
            assert indicator.dim2 is None and indicator.dim3 is None
        if indicator.dim2 is None:
            assert indicator.dim3 is None
