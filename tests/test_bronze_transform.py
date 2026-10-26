from datetime import datetime, timezone

import pandas as pd
import pytest

from data_health_ftw.bronze.transform import (
    add_ingestion_metadata,
    build_raw_frame,
    cast_to_string,
    extract_records,
)


def test_extract_records_raises_on_empty_payload():
    with pytest.raises(ValueError):
        extract_records({"value": []}, "BP_04")


def test_build_raw_frame_concatenates_indicators():
    payloads = {
        "A": {"value": [{"SpatialDim": "BRA", "NumericValue": 1.0}]},
        "B": {"value": [{"SpatialDim": "ARG", "NumericValue": 2.0}]},
    }
    assert len(build_raw_frame(payloads)) == 2


def test_add_ingestion_metadata_is_deterministic():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = add_ingestion_metadata(pd.DataFrame({"x": [1]}), "http://x", ts)
    assert result.loc[0, "_ingested_at"] == "2026-01-01T00:00:00+00:00"
    assert result.loc[0, "_ingestion_date"] == "2026-01-01"


def test_cast_to_string_keeps_nulls_as_missing():
    result = cast_to_string(pd.DataFrame({"x": [1.5, None], "y": ["a", None]}))
    assert result.loc[0, "x"] == "1.5"
    assert pd.isna(result.loc[1, "x"])
    assert pd.isna(result.loc[1, "y"])
