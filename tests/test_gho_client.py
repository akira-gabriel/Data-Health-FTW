import pytest
import requests

from data_health_ftw.bronze import gho_client
from data_health_ftw.common.config import Indicator


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


def test_build_indicator_filter_selects_published_country_rows():
    result = gho_client.build_indicator_filter("SDGPM25")
    assert "IND_CODE eq 'SDGPM25'" in result
    assert "DIM_GEO_CODE_TYPE eq 'COUNTRY'" in result
    assert "DIM_PUBLISH_STATE_CODE eq 'PUBLISHED'" in result


def test_fetch_all_pages_pages_with_skip_until_an_empty_page():
    session = FakeSession([
        FakeResponse({"value": [{"a": 1}, {"a": 2}]}),
        FakeResponse({"value": [{"a": 3}]}),
        FakeResponse({"value": []}),
    ])
    rows = gho_client.fetch_all_pages(session, "http://x/E", {"$filter": "f"}, 5, page_size=2)
    assert [r["a"] for r in rows] == [1, 2, 3]
    assert [c[1]["$skip"] for c in session.calls] == [0, 2, 3]


def test_fetch_all_pages_does_not_stop_when_server_caps_page_size():
    session = FakeSession([
        FakeResponse({"value": [{"a": 1}]}),  # asked for 5000, server returned 1
        FakeResponse({"value": [{"a": 2}]}),
        FakeResponse({"value": []}),
    ])
    rows = gho_client.fetch_all_pages(session, "http://x/E", {}, 5, page_size=5000)
    assert len(rows) == 2


def test_fetch_all_pages_follows_next_link():
    session = FakeSession([
        FakeResponse({"value": [{"a": 1}], "@odata.nextLink": "http://x/E?page=2"}),
        FakeResponse({"value": [{"a": 2}]}),
        FakeResponse({"value": []}),
    ])
    rows = gho_client.fetch_all_pages(session, "http://x/E", {}, 5, page_size=10)
    assert len(rows) == 2
    assert session.calls[1] == ("http://x/E?page=2", None)


def test_fetch_indicator_payload_queries_the_indicator_entity():
    session = FakeSession([FakeResponse({"value": [{"a": 1}]}), FakeResponse({"value": []})])
    indicator = Indicator("NCD_BMI_30C", "obesity_prev", "TOTAL", "Y_GE18", entity="RELAY_NUTRITION")
    payload = gho_client.fetch_indicator_payload(session, "http://x/DATA_/", indicator, 5, 100)
    assert payload == {"value": [{"a": 1}]}
    assert session.calls[0][0] == "http://x/DATA_/RELAY_NUTRITION"


def test_get_json_retries_retryable_status(monkeypatch):
    monkeypatch.setattr(gho_client.time, "sleep", lambda _: None)
    session = FakeSession([FakeResponse({}, 500), FakeResponse({"value": [1]})])
    assert gho_client._get_json(session, "http://x", None, 5)["value"] == [1]


def test_get_json_raises_after_last_attempt(monkeypatch):
    monkeypatch.setattr(gho_client.time, "sleep", lambda _: None)
    session = FakeSession([FakeResponse({}, 503)] * 3)
    with pytest.raises(requests.HTTPError):
        gho_client._get_json(session, "http://x", None, 5, max_attempts=3)