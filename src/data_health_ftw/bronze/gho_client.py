import logging
import time

import requests

logger = logging.getLogger(__name__)


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def fetch_indicator_payload(
    session: requests.Session,
    base_url: str,
    code: str,
    timeout: int,
    max_attempts: int = 4,
    backoff_seconds: float = 2.0,
) -> dict:
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(f"{base_url}{code}", timeout=timeout)
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_attempts:
                raise requests.HTTPError(f"retryable status {response.status_code}")
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == max_attempts:
                raise
            wait = backoff_seconds * 2 ** (attempt - 1)
            logger.warning("Retry %d/%d for %s in %.0fs", attempt, max_attempts, code, wait)
            time.sleep(wait)

def fetch_all_payloads(
    codes: list[str], base_url: str, timeout: int, pause_seconds: float
) -> tuple[dict[str, dict], list[str]]:

    payloads: dict[str, dict] = {}
    failed: list[str] = []

    with requests.Session() as session:
        for code in codes:
            try:
                payloads[code] = fetch_indicator_payload(session, base_url, code, timeout)
            except requests.RequestException:
                logger.exception("Request failed for indicator %s", code)
                failed.append(code)
            time.sleep(pause_seconds)

    return payloads, failed