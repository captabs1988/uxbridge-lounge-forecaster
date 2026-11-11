"""
data_cleaning.py: Phase 2 as a reusable script.

This is the SAME cleaning you did step by step in notebooks/02_data_cleaning.ipynb,
turned into functions so it runs top to bottom every time (no notebook traps).

Run it from the project folder:
    python -m src.data_cleaning
Input : data/raw/uxbridge_lounge_sales_raw.csv  (never changed)
Output: data/processed/uxbridge_lounge_sales_clean.csv + docs/cleaning_log.csv
"""
import pandas as pd  # tables of data

from src import config  # all paths and settings

# Typos found during profiling (Step 2.2), mapped to the official menu name
ITEM_TYPOS = {
    "butter chiken": "Butter Chicken", "masala dosaa": "Masala Dosa",
    "chole bature": "Chole Bhature", "paav bhaji": "Pav Bhaji",
    "idly sambar": "Idli Sambar", "dhoklaa": "Dhokla",
    "kingfisher": "Kingfisher Lager",
}


def fix_structure(df, log):
    """Stage A: duplicates, dates, day_of_week, order."""
    before = len(df)  # rows before
    df = df.drop_duplicates()  # remove exact copies
    log.append(("Removed exact duplicate rows", before - len(df)))

    df["date"] = (pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")  # 2025-02-27
                  .fillna(pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce"))  # 27/02/2025
                  .fillna(pd.to_datetime(df["date"], format="%d %b %Y", errors="coerce")))  # 27 Feb 2025
    assert df["date"].isna().sum() == 0, "Some dates could not be read"  # tripwire

    real_day = df["date"].dt.day_name()  # true day from the date
    log.append(("Corrected day_of_week to match date", int((real_day != df["day_of_week"]).sum())))
    df["day_of_week"] = real_day  # replace with the truth

    df = df.sort_values(["date", "record_id"]).reset_index(drop=True)  # oldest first
    log.append(("Sorted rows by date", len(df)))
    return df


def fix_text(df, log):
    """Stage B: make every text column use one official spelling."""
    lookup = {name.lower(): name for name in config.MENU}  # "butter chicken" -> "Butter Chicken"
    lookup.update(ITEM_TYPOS)  # add the known typos
    old = df["item_name"].copy()  # remember old values to count changes
    df["item_name"] = df["item_name"].str.strip().str.lower().map(lookup)  # strip, lower, translate
    assert df["item_name"].isna().sum() == 0, "Unknown item name found"  # every name must be on the menu
    log.append(("Standardised item names to menu", int((old != df["item_name"]).sum())))

    old = df["category"].copy()
    df["category"] = df["category"].str.strip().str.lower().map({"food": "Food", "drink": "Drink", "drinks": "Drink"})
    assert df["category"].isna().sum() == 0, "Unknown category found"
    log.append(("Standardised category", int((old != df["category"]).sum())))

    old = df["weather"].copy()
    df["weather"] = df["weather"].str.strip().str.title().replace({"Rain": "Rainy"})  # blanks stay blank for Stage C
    log.append(("Standardised weather spelling", int(((old != df["weather"]) & old.notna()).sum())))

    df["festival"] = (df["festival"].fillna("None").str.strip()
                      .replace({"none": "None", "-": "None", "No festival": "None"}))  # 4 versions of "no festival"
    log.append(("Standardised 'no festival' to None", int((df["festival"] == "None").sum())))

    df["local_event"] = df["local_event"].fillna("None")  # blank = no event
    log.append(("Filled blank local_event with None", int((df["local_event"] == "None").sum())))

    old = df["stockout"].copy()
    df["stockout"] = df["stockout"].astype(str).str.strip().str.upper().map(
        {"YES": "Yes", "Y": "Yes", "TRUE": "Yes", "NO": "No", "N": "No", "FALSE": "No"})  # 8 versions -> 2
    assert df["stockout"].isna().sum() == 0, "Unknown stockout value found"
    log.append(("Standardised stockout to Yes/No", int((old != df["stockout"]).sum())))
    return df


def fix_numbers(df, log):
    """Stage C: price, temperature and weather, using the 'team huddle' per day."""
    old = df["price_gbp"].astype(str)  # look at prices as text
    df["price_gbp"] = pd.to_numeric(old.str.replace("£", "", regex=False))  # remove £, make number
    log.append(("Removed £ sign from price", int(old.str.contains("£").sum())))

    temp = pd.to_numeric(df["temperature_c"], errors="coerce")  # "unknown", "-" become blank
    day_temp = temp.groupby(df["date"]).transform("median")  # each day votes: the middle value wins
    log.append(("Fixed temperature (Fahrenheit, text, blanks) using daily median", int((temp != day_temp).sum())))
    df["temperature_c"] = day_temp

    blanks = int(df["weather"].isna().sum())
    df["weather"] = df.groupby("date")["weather"].transform(lambda s: s.mode().iloc[0])  # most common that day
    log.append(("Filled blank weather using most common weather that day", blanks))
    return df


def fix_quantities(df, log):
    """Stage D: negatives, impossible sales, blanks, waste. ORDER MATTERS."""
    neg = (df["sold_qty"] < 0) | (df["wasted_qty"] < 0)
    df["sold_qty"] = df["sold_qty"].abs()  # -39 -> 39 (typed minus sign)
    df["wasted_qty"] = df["wasted_qty"].abs()
    log.append(("Removed minus sign from negative quantities", int(neg.sum())))

    impossible = df["sold_qty"] > df["prepared_qty"]  # can't sell more than you made
    df.loc[impossible, "sold_qty"] = float("nan")  # blank it, don't guess
    log.append(("Blanked impossible sold_qty (more than prepared)", int(impossible.sum())))

    fresh = df["shelf_life"] == "Same day"
    fill = fresh & df["sold_qty"].isna()
    df.loc[fill, "sold_qty"] = df.loc[fill, "prepared_qty"] - df.loc[fill, "wasted_qty"]  # sold = prepared - wasted
    log.append(("Filled blank sold_qty for fresh items (prepared - wasted)", int(fill.sum())))

    drop = df["sold_qty"].isna()  # bottles we can't rebuild
    df = df[~drop].reset_index(drop=True)
    log.append(("Dropped bottle rows with unknown sold_qty", int(drop.sum())))
    df["sold_qty"] = df["sold_qty"].astype(int)  # whole numbers

    fresh = df["shelf_life"] == "Same day"
    correct_waste = df["prepared_qty"] - df["sold_qty"]
    bad = fresh & (df["wasted_qty"] != correct_waste)
    df.loc[bad, "wasted_qty"] = correct_waste[bad]  # trust the till, fix the hand-written waste
    log.append(("Corrected wasted_qty to prepared - sold (fresh items)", int(bad.sum())))
    return df


def validate(df):
    """Stage E: business rules the clean data MUST pass. Returns {rule: True/False}."""
    qty = ["prepared_qty", "sold_qty", "wasted_qty"]
    fresh = df["shelf_life"] == "Same day"
    return {
        "No duplicate rows": df.duplicated().sum() == 0,
        "record_id is unique": df["record_id"].is_unique,
        "No blanks anywhere": df.isna().sum().sum() == 0,
        "Rows in date order": df["date"].is_monotonic_increasing,
        "Dates within Jan 2025 to Jun 2026": df["date"].between("2025-01-01", "2026-06-30").all(),
        "day_of_week matches date": (df["date"].dt.day_name() == df["day_of_week"]).all(),
        "24 menu items only": set(df["item_name"]) == set(config.MENU),
        "Category is Food or Drink": set(df["category"]) == {"Food", "Drink"},
        "Weather has 4 valid types": set(df["weather"]) <= set(config.WEATHER_TYPES),
        "Stockout is Yes or No": set(df["stockout"]) == {"Yes", "No"},
        "One price per item": df.groupby("item_name")["price_gbp"].nunique().max() == 1,
        "One temperature per day": df.groupby("date")["temperature_c"].nunique().max() == 1,
        "Temperature in London range": df["temperature_c"].between(-10, 35).all(),
        "No negative quantities": (df[qty] >= 0).all().all(),
        "Sold never more than prepared": (df["sold_qty"] <= df["prepared_qty"]).all(),
        "Fresh waste adds up": ((df["prepared_qty"] - df["sold_qty"] == df["wasted_qty"]) | ~fresh).all(),
        "Stockout Yes means sold out": ((df["stockout"] == "No") | (df["sold_qty"] == df["prepared_qty"])).all(),
        "At least 1,000 rows": len(df) >= 1000,
    }


def clean_data(raw):
    """Run every stage in the right order. Returns (clean table, log table)."""
    log = []  # list of (step, rows affected)
    df = raw.copy()  # never touch the raw table
    df = fix_structure(df, log)
    df = fix_text(df, log)
    df = fix_numbers(df, log)
    df = fix_quantities(df, log)
    return df, pd.DataFrame(log, columns=["step", "rows_affected"])


def load_clean():
    """Read the clean CSV with the date column as real dates (used by other files).

    keep_default_na=False: stops pandas turning the WORD "None" (no festival) into a blank.
    Safe here because the clean file has no blanks at all (validation rule).
    """
    return pd.read_csv(config.CLEAN_SALES_FILE, parse_dates=["date"], keep_default_na=False)


def main():
    raw = pd.read_csv(config.RAW_SALES_FILE)  # read the messy file
    clean, log = clean_data(raw)  # clean it
    checks = validate(clean)  # prove it
    for rule, passed in checks.items():
        print(("PASS" if passed else "FAIL"), rule)
    assert all(checks.values()), "Validation failed, clean file NOT saved"  # tripwire before saving
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(config.CLEAN_SALES_FILE, index=False)  # save clean data
    log.to_csv(config.CLEANING_LOG_FILE, index=False)  # save evidence
    print(f"\nSaved {len(clean):,} clean rows ({len(raw):,} raw) and {len(log)} log steps.")


if __name__ == "__main__":  # only runs when you call this file directly
    main()
