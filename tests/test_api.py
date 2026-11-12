"""Tests for Phase 8.3: the API answers correctly and rejects bad requests."""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)  # a fake browser that talks to the API without starting a server


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_menu_has_24_items():
    assert len(client.get("/menu").json()) == 24


def test_forecast_ok():
    r = client.post("/forecast", json={"date": "2026-07-04", "weather": "Sunny", "temperature_c": 22})
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 24 and body["summary"]["total_recommended_prep"] > 0


def test_forecast_rejects_bad_weather():
    r = client.post("/forecast", json={"date": "2026-07-04", "weather": "Foggy", "temperature_c": 22})
    assert r.status_code == 422  # 422 = "your request doesn't make sense"


def test_forecast_rejects_crazy_temperature():
    r = client.post("/forecast", json={"date": "2026-07-04", "weather": "Sunny", "temperature_c": 78})
    assert r.status_code == 422


def test_home_redirects_to_docs():
    r = client.get("/", follow_redirects=False)  # don't follow, just check where it points
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"


def test_vercel_entrypoint_exposes_app():
    import index  # the file Vercel loads
    assert index.app is app  # same API object, so Vercel serves exactly what we tested
