import logging
import time
from collections.abc import Sequence

import pandas as pd
import requests

from data_health_ftw.common.config import Indicator

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def build_indicator_filter(code: str) -> str:
    return (
        f"IND_CODE eq '{code}' and DIM_GEO_CODE_TYPE eq 'COUNTRY'"
        " and DIM_PUBLISH_STATE_CODE eq 'PUBLISHED'"
    )


def _get_json(
    session: requests.Session,
    url: str,
    params: dict | None,
    timeout: int,
    max_attempts: int = 4,
    backoff_seconds: float = 2.0,
) -> dict:
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_attempts:
                raise requests.HTTPError(f"retryable status {response.status_code}")
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == max_attempts:
                raise
            wait = backoff_seconds * 2 ** (attempt - 1)
            logger.warning("Retry %d/%d for %s in %.0fs", attempt, max_attempts, url, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_all_pages(
    session: requests.Session,
    url: str,
    params: dict,
    timeout: int,
    page_size: int,
) -> list[dict]:
    records: list[dict] = []
    skip = 0
    next_url: str | None = url
    next_params: dict | None = {**params, "$top": page_size, "$skip": 0}
    while next_url:
        payload = _get_json(session, next_url, next_params, timeout)
        batch = payload.get("value", [])
        if not batch:
            break
        records.extend(batch)
        next_url = payload.get("@odata.nextLink")
        if next_url:
            next_params = None
        else:
            skip += len(batch)
            next_url = url
            next_params = {**params, "$top": page_size, "$skip": skip}
    return records


def fetch_indicator_payload(
    session: requests.Session,
    base_url: str,
    indicator: Indicator,
    timeout: int,
    page_size: int,
) -> dict:
    records = fetch_all_pages(
        session,
        f"{base_url}{indicator.entity}",
        {"$filter": build_indicator_filter(indicator.code)},
        timeout,
        page_size,
    )
    return {"value": records}


def fetch_all_payloads(
    indicators: Sequence[Indicator],
    base_url: str,
    timeout: int,
    pause_seconds: float,
    page_size: int,
) -> tuple[dict[str, dict], list[str]]:
    payloads: dict[str, dict] = {}
    failed: list[str] = []

    with requests.Session() as session:
        for indicator in indicators:
            try:
                payloads[indicator.code] = fetch_indicator_payload(
                    session, base_url, indicator, timeout, page_size
                )
            except requests.RequestException:
                logger.exception("Request failed for indicator %s", indicator.code)
                failed.append(indicator.code)
            time.sleep(pause_seconds)

    return payloads, failed


def fetch_country_reference(
    refmart_base_url: str, timeout: int, page_size: int
) -> pd.DataFrame:
    """M49 -> ISO-3 and WHO region lookup (REFMART.REF_COUNTRY)."""
    with requests.Session() as session:
        records = fetch_all_pages(
            session,
            f"{refmart_base_url}REF_COUNTRY",
            {"$select": "GEO_M49_CODE,CODE_ISO_3,GRP_WHO_REGION"},
            timeout,
            page_size,
        )
    if not records:
        raise ValueError("REFMART.REF_COUNTRY returned no rows")
    return pd.DataFrame(records)