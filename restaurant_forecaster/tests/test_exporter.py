from src import exporter


class _FakeResponse:
    def __init__(self, status_code=200, text=""):
        self.status_code = status_code
        self.text = text


def test_push_projections_success(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _FakeResponse(status_code=200)

    monkeypatch.setattr(exporter.requests, "post", fake_post)

    result = exporter.push_projections(
        endpoint="https://example.com/webhook",
        api_key="secret-key",
        daily_projections=[{"date": "2024-05-01", "projected_sales": 1000.0}],
        accuracy_metrics={"sales": {}},
    )

    assert result.success
    assert result.status_code == 200
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["json"]["daily_projections"][0]["date"] == "2024-05-01"


def test_push_projections_retries_then_fails(monkeypatch):
    call_count = {"n": 0}

    def fake_post(url, json=None, headers=None, timeout=None):
        call_count["n"] += 1
        return _FakeResponse(status_code=500, text="server error")

    monkeypatch.setattr(exporter.requests, "post", fake_post)

    result = exporter.push_projections(
        endpoint="https://example.com/webhook",
        api_key="secret-key",
        daily_projections=[],
        accuracy_metrics={},
        max_retries=2,
    )

    assert not result.success
    assert call_count["n"] == 2
    assert "500" in result.error
