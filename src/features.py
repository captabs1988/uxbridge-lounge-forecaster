"""
features.py: Phase 4, feature engineering.

A "feature" is a clue the model uses to make a prediction.
We turn raw columns into clues a model can learn from.

DESIGN DECISION (good to explain to an assessor):
The model only uses clues that are KNOWN BEFORE the day starts
(date, menu item, weather forecast, festival, event).
It does NOT use yesterday's sales. That means the chef can forecast
ANY future date in the app, not just tomorrow.
Last week's sales are still used, but only for the BASELINE (Phase 5).
"""
import pandas as pd  # tables

from src import config  # settings

START_DATE = pd.Timestamp("2025-01-01")  # day 0 for the trend clue

CATEGORICAL = ["item_name", "category", "cuisine", "day_of_week", "weather", "festival"]  # word clues
NUMERIC = ["month", "is_weekend", "days_since_start", "temperature_c",
           "is_cricket", "is_festival"]  # number clues
# price_gbp is left out on purpose: each item has ONE fixed price, so it only repeats item_name
# and would split the explainability credit between two clues that mean the same thing.
FEATURES = CATEGORICAL + NUMERIC  # everything the model sees
TARGET = "sold_qty"  # what we predict


def add_features(df):
    """Add the engineered clues to a table that has date, item and conditions."""
    out = df.copy()  # never change the input table
    out["date"] = pd.to_datetime(out["date"])  # make sure it's a real date
    out["month"] = out["date"].dt.month  # 1 to 12: seasons
    out["is_weekend"] = out["date"].dt.dayofweek.isin([4, 5]).astype(int)  # Fri and Sat are the busy nights
    out["days_since_start"] = (out["date"] - START_DATE).dt.days  # lets the model learn business growth
    out["is_cricket"] = (out["local_event"] == "India cricket match").astype(int)  # 1 = match on TV
    out["is_festival"] = (out["festival"] != "None").astype(int)  # 1 = any festival
    return out


def add_business_columns(df):
    """Phase 4.4: money columns used in analysis and the business simulation."""
    out = df.copy()
    out["revenue_gbp"] = out["sold_qty"] * out["price_gbp"]  # money taken
    out["food_cost_gbp"] = out["prepared_qty"] * out["unit_cost_gbp"]  # cost of everything prepared
    out["waste_cost_gbp"] = out["wasted_qty"] * out["unit_cost_gbp"]  # money thrown in the bin
    out["gross_profit_gbp"] = out["revenue_gbp"] - out["food_cost_gbp"]  # what's left after food cost
    return out


def add_baseline_lag(df):
    """Baseline clue: what this item sold on the same weekday last week (7 days ago)."""
    out = df.sort_values(["item_name", "date"]).copy()  # each item in date order
    last_week = out[["item_name", "date", "sold_qty"]].copy()
    last_week["date"] = last_week["date"] + pd.Timedelta(days=7)  # move last week's sales forward 7 days
    last_week = last_week.rename(columns={"sold_qty": "sold_last_week"})
    out = out.merge(last_week, on=["item_name", "date"], how="left")  # line them up with this week
    return out.sort_values(["date", "item_name"]).reset_index(drop=True)


def time_split(df):
    """Phase 4.5: train on the PAST, test on the FUTURE. No shuffling, ever, for time data."""
    train = df[df["date"] <= config.TRAIN_END].reset_index(drop=True)  # Jan 2025 to Feb 2026
    test = df[df["date"] >= config.TEST_START].reset_index(drop=True)  # Mar 2026 to Jun 2026
    return train, test


def build_future_rows(date, weather, temperature_c, festival="None", local_event="None", menu_df=None):
    """Create one row per menu item for a day we want to forecast (used by the app and API)."""
    rows = menu_df.copy()  # the 24 menu items with their category, cuisine, price
    rows["date"] = pd.Timestamp(date)
    rows["day_of_week"] = rows["date"].dt.day_name()
    rows["weather"] = weather
    rows["temperature_c"] = float(temperature_c)
    rows["festival"] = festival
    rows["local_event"] = local_event
    return add_features(rows)
