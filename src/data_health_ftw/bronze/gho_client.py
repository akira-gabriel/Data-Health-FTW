import logging
import time

import requests

logger = logging.getLogger(__name__)


def fetch_indicator_payload(
    session: requests.Session, base_url: str, code: str, timeout: int
) -> dict:
    response = session.get(f"{base_url}{code}", timeout=timeout)
    response.raise_for_status()
    return response.json()


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