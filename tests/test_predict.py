"""Tests for Phase 8.2: the forecast function must give sensible, safe answers."""
import time

import pytest

from src import config
from src.predict import forecast_day, summary

SUNNY_SATURDAY = dict(date="2026-07-04", weather="Sunny", temperature_c=22)


@pytest.fixture(scope="module")
def table():
    return forecast_day(**SUNNY_SATURDAY)


def test_one_row_per_menu_item(table):
    assert list(table["item_name"]) == config.MENU


def test_no_negative_numbers(table):
    numbers = ["forecast", "likely_low", "likely_high", "recommended_prep", "expected_revenue_gbp"]
    assert (table[numbers] >= 0).all().all()


def test_range_contains_forecast(table):
    assert (table["likely_low"] <= table["forecast"]).all()
    assert (table["forecast"] <= table["likely_high"]).all()


def test_prep_is_whole_and_at_least_forecast(table):
    assert (table["recommended_prep"] >= table["forecast"]).all()  # never prep less than expected demand


def test_risk_is_a_percentage(table):
    assert table["stockout_risk_pct"].between(0, 100).all()


def test_same_input_same_output():
    a, b = forecast_day(**SUNNY_SATURDAY), forecast_day(**SUNNY_SATURDAY)
    assert a.equals(b)  # reproducible: no hidden randomness at prediction time


def test_business_logic_cricket_sells_more_beer():
    normal = forecast_day(**SUNNY_SATURDAY).set_index("item_name")
    match = forecast_day(**SUNNY_SATURDAY, local_event="India cricket match").set_index("item_name")
    assert match.loc["Kingfisher Lager", "forecast"] > normal.loc["Kingfisher Lager", "forecast"]


def test_business_logic_saturday_busier_than_monday():
    sat = forecast_day("2026-07-04", "Cloudy", 15)["forecast"].sum()
    mon = forecast_day("2026-07-06", "Cloudy", 15)["forecast"].sum()
    assert sat > mon


def test_bigger_buffer_means_more_prep():
    small = forecast_day(**SUNNY_SATURDAY, service_level=0.5)["recommended_prep"].sum()
    big = forecast_day(**SUNNY_SATURDAY, service_level=0.9)["recommended_prep"].sum()
    assert big > small


@pytest.mark.parametrize("bad", [dict(weather="Foggy"), dict(festival="Eid"), dict(temperature_c=78)])
def test_bad_input_is_rejected(bad):
    args = {**SUNNY_SATURDAY, **bad}
    with pytest.raises(ValueError):  # must fail loudly, never guess
        forecast_day(**args)


def test_chef_plan_comparison():
    t = forecast_day(**SUNNY_SATURDAY, planned_prep={"Masala Dosa": 5})
    dosa = t.set_index("item_name").loc["Masala Dosa"]
    assert dosa["chef_planned_prep"] == 5
    assert dosa["chef_stockout_risk_pct"] > dosa["stockout_risk_pct"]  # 5 dosas on a Saturday is risky


def test_fast_enough():
    forecast_day(**SUNNY_SATURDAY)  # warm up (loads models)
    start = time.perf_counter()
    forecast_day(**SUNNY_SATURDAY)
    assert time.perf_counter() - start < 0.5  # under half a second


def test_summary_totals(table):
    s = summary(table)
    assert s["total_recommended_prep"] == table["recommended_prep"].sum()
