import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def extract_records(payload: dict, code: str) -> list[dict]:
    records = payload.get("value")
    if not records:
        raise ValueError(f"Empty response for indicator {code}")
    return records


def build_raw_frame(
    payloads: dict[str, dict], column_map: dict[str, str] | None = None
) -> pd.DataFrame:
    frames = [
        pd.DataFrame(extract_records(payload, code))
        for code, payload in payloads.items()
    ]

    frames = [frame for frame in frames if not frame.empty]
    raw_df = pd.concat(frames, ignore_index=True)
    if column_map:
        raw_df = raw_df[list(column_map)].rename(columns=column_map)
    return raw_df


def _normalize_m49(codes: pd.Series) -> pd.Series:
    """M49 codes can be padded ("004") in one source and not in the other ("4")."""
    return codes.astype("string").str.strip().str.lstrip("0")


def attach_country_reference(df: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    reference = (
        countries.assign(_m49=_normalize_m49(countries["GEO_M49_CODE"]))
        .loc[:, ["_m49", "CODE_ISO_3", "GRP_WHO_REGION"]]
        .drop_duplicates("_m49")
    )
    merged = df.assign(_m49=_normalize_m49(df["GeoM49"])).merge(reference, on="_m49", how="left")

    unmapped = merged["CODE_ISO_3"].isna()
    if unmapped.any():
        logger.warning(
            "Dropping %d rows with M49 codes missing from REF_COUNTRY: %s",
            int(unmapped.sum()),
            sorted(merged.loc[unmapped, "GeoM49"].dropna().unique()),
        )
    return (
        merged.loc[~unmapped]
        .rename(columns={"CODE_ISO_3": "SpatialDim", "GRP_WHO_REGION": "ParentLocationCode"})
        .drop(columns=["_m49", "GeoM49"])
        .reset_index(drop=True)
    )


def add_ingestion_metadata(
    df: pd.DataFrame, source_url: str, ingested_at: datetime
) -> pd.DataFrame:
    return df.assign(
        _ingested_at=ingested_at.isoformat(),
        _ingestion_date=ingested_at.date().isoformat(),
        _source_url=source_url,
    )


def cast_to_string(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(str).where(df.notna(), None)