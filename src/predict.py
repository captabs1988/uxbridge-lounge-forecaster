"""
predict.py: Phase 8.2, the ONE function the app and API call to get a forecast.

    from src.predict import forecast_day
    table = forecast_day("2026-07-04", weather="Sunny", temperature_c=22)

Returns one row per menu item with: forecast, likely range, recommended prep,
stockout risk, expected revenue and expected waste cost.
"""
import json  # read metadata
from functools import lru_cache  # load the models once, then reuse them

import joblib  # load saved models
import numpy as np  # maths
import pandas as pd  # tables

from src import config
from src.features import FEATURES, build_future_rows


@lru_cache(maxsize=1)  # first call loads from disk, every later call is instant
def load_artifacts():
    """Load the 2 saved models and the metadata (menu, settings, calibrated ratios)."""
    meta = json.loads(config.METADATA_FILE.read_text())
    return {
        "forecast": joblib.load(config.FORECAST_MODEL_FILE),
        "stockout": joblib.load(config.STOCKOUT_MODEL_FILE),
        "meta": meta,
        "menu": pd.DataFrame(meta["menu"]),
    }


def validate_inputs(weather, temperature_c, festival, local_event):
    """Reject inputs the model has never seen, with a clear message (fail loudly, not silently)."""
    if weather not in config.WEATHER_TYPES:
        raise ValueError(f"weather must be one of {config.WEATHER_TYPES}, got '{weather}'")
    if festival not in config.FESTIVALS:
        raise ValueError(f"festival must be one of {config.FESTIVALS}, got '{festival}'")
    if local_event not in config.LOCAL_EVENTS:
        raise ValueError(f"local_event must be one of {config.LOCAL_EVENTS}, got '{local_event}'")
    if not -10 <= float(temperature_c) <= 35:
        raise ValueError(f"temperature_c must be between -10 and 35 (London range), got {temperature_c}")


def ratio(level):
    """Calibrated ratio for a level, e.g. ratio(0.9) = 1.12 means demand is below 112% of the forecast on 90% of days."""
    table = load_artifacts()["meta"]["ratio_table"]
    return table[str(round(min(max(level, 0.05), 0.95) * 20) / 20)]  # snap to the nearest 5%


def forecast_day(date, weather, temperature_c, festival="None", local_event="None",
                 planned_prep=None, service_level=config.SERVICE_LEVEL):
    """Forecast every menu item for one day.

    planned_prep : optional {item_name: portions} the chef plans to make,
                   so we can show the stockout risk of HIS plan next to ours.
    service_level: buffer for fresh items (0.5 = no buffer, 0.9 = big buffer). Bottles always use 0.9.
    """
    validate_inputs(weather, temperature_c, festival, local_event)
    art = load_artifacts()
    rows = build_future_rows(date, weather, temperature_c, festival, local_event, art["menu"])
    X = rows[FEATURES]  # exactly the clues the model was trained on

    forecast = np.maximum(art["forecast"].predict(X), 0)  # expected demand (never below 0)
    low = forecast * ratio(config.INTERVAL_LOW)  # 80% likely range, bottom
    high = forecast * ratio(config.INTERVAL_HIGH)  # 80% likely range, top
    fresh = (rows["shelf_life"] == "Same day").to_numpy()  # freshly made = unsold goes in the bin
    level = np.where(fresh, ratio(service_level), ratio(config.BOTTLE_SERVICE_LEVEL))  # bottles: generous
    rec = np.ceil(forecast * level)  # recommended prep, whole portions

    def risk(prep):  # probability of running out if we prep this many
        clf_X = X.assign(prepared_qty=prep, prep_vs_forecast=prep / np.maximum(forecast, 0.5))
        return art["stockout"].predict_proba(clf_X)[:, 1]

    out = pd.DataFrame({
        "item_name": rows["item_name"],
        "category": rows["category"],
        "cuisine": rows["cuisine"],
        "forecast": forecast.round(1),
        "likely_low": np.floor(low).astype(int),
        "likely_high": np.ceil(high).astype(int),
        "recommended_prep": rec.astype(int),
        "stockout_risk_pct": (100 * risk(rec)).round(0).astype(int),
        "stockout_alert": risk(rec) >= art["meta"]["stockout_alert_threshold"],  # True = warn the chef
        "expected_revenue_gbp": (forecast * rows["price_gbp"]).round(2),
        "expected_waste_cost_gbp": np.where(fresh, np.maximum(rec - forecast, 0) * rows["unit_cost_gbp"], 0).round(2),
    })
    if planned_prep:  # compare the chef's own plan
        chef = rows["item_name"].map(planned_prep).fillna(pd.Series(rec, index=rows.index)).to_numpy(dtype=float)
        out["chef_planned_prep"] = chef.astype(int)
        out["chef_stockout_risk_pct"] = (100 * risk(chef)).round(0).astype(int)
    return out


def summary(table):
    """Totals for the dashboard header."""
    return {
        "total_forecast_portions": round(float(table["forecast"].sum()), 1),
        "total_recommended_prep": int(table["recommended_prep"].sum()),
        "expected_revenue_gbp": round(float(table["expected_revenue_gbp"].sum()), 2),
        "expected_waste_cost_gbp": round(float(table["expected_waste_cost_gbp"].sum()), 2),
        "high_risk_items": table.loc[table["stockout_alert"], "item_name"].tolist(),
    }


if __name__ == "__main__":  # quick manual test: python -m src.predict
    t = forecast_day("2026-07-04", weather="Sunny", temperature_c=22, local_event="India cricket match")
    print(t.to_string(index=False))
    print(summary(t))
