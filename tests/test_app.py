"""Tests for Phase 9: the Streamlit app starts, shows numbers, and reacts to the chef's inputs."""
from pathlib import Path  # file paths that work on Windows and Mac

from streamlit.testing.v1 import AppTest  # runs the app in memory, no browser needed

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")  # full path to the app


def run_app():
    """Start the app once and return it (60s timeout because the first run loads the models)."""
    return AppTest.from_file(APP, default_timeout=60).run()


def test_app_starts_without_errors():
    at = run_app()
    assert not at.exception  # no red error box on the page


def test_app_shows_kpi_cards():
    at = run_app()
    labels = [m.label for m in at.metric]  # every metric card on the page
    assert any("Recommended prep" in label for label in labels)
    assert any("waste" in label.lower() for label in labels)


def test_festival_changes_forecast():
    at = run_app()
    normal = at.metric[0].value  # expected portions on a normal day
    festival_box = [s for s in at.sidebar.selectbox if "Festival" in s.label][0]  # find the festival dropdown
    festival_box.set_value("Diwali").run()  # pretend the chef picked Diwali
    assert not at.exception
    assert at.metric[0].value != normal  # the forecast must react to the festival


def test_identical_plan_has_same_waste_as_ai():
    at = run_app()
    chef_waste = [m for m in at.metric if m.label == "Your expected waste"][0]
    assert chef_waste.delta.startswith("£0.00")  # untouched plan = same as AI, so no difference
