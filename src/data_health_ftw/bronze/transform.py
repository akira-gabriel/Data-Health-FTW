from datetime import datetime

import pandas as pd


def extract_records(payload: dict, code: str) -> list[dict]:
    records = payload.get("value")
    if not records:
        raise ValueError(f"Empty response for indicator {code}")
    return records


def build_raw_frame(payloads: dict[str, dict]) -> pd.DataFrame:
    frames = [
        pd.DataFrame(extract_records(payload, code))
        for code, payload in payloads.items()
    ]

    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True)


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