"""
generate_data.py: Phase 1 as a script (same code as notebooks/01_generate_data.ipynb).
Run: python -m src.generate_data   -> rebuilds data/raw/uxbridge_lounge_sales_raw.csv exactly (seed 42).
"""
# %% [markdown]
# # Uxbridge Lounge: Messy Dataset Generator
#
# **Project:** Food and Drink Demand and Waste Forecaster for Uxbridge Lounge (Indian restaurant, Uxbridge)
#
# **What this notebook does:** creates a realistic, AI-generated daily sales dataset (Jan 2025 to Jun 2026),
# then deliberately adds real-world data problems, so the next phase (data cleaning) is a genuine exercise.
#
# **Why generated data:** real company data would need customer consent (GDPR). This data contains
# no personal information at all, and the fixed random seed makes it 100% reproducible.
#
# **Calibrated on real public data:** waste rates are anchored to the public Kaggle dataset
# `food_wastage_data.csv` (1,782 rows, 1,618 after removing duplicates). Key benchmarks used:
# average waste = 6.9% of food prepared; high-priced items waste more (8.8%) than moderate (5.8%)
# or low-priced (5.5%); bigger, busier events waste a bit more. See Section 4b for the check.
#
# ## Data dictionary (what each column means)
# - `record_id`: unique ID for each row
# - `date`: the trading day
# - `day_of_week`: Monday to Sunday
# - `item_name`: menu item
# - `category`: Food or Drink
# - `cuisine`: Punjabi, Gujarati, Marathi, South Indian, or Bar for alcohol
# - `price_gbp`: selling price per portion or drink
# - `price_tier`: Low (under £6), Moderate (£6 to £9.99), High (£10+), matching the Kaggle benchmark
# - `unit_cost_gbp`: cost to make or buy one portion or drink
# - `shelf_life`: "Same day" (freshly made, unsold = waste) or "Long" (bottled, unsold goes back to stock)
# - `weather`: Sunny, Cloudy, Rainy or Snow
# - `temperature_c`: average temperature in Celsius
# - `festival`: festival happening that day (Diwali, Navratri, etc.) or None
# - `local_event`: e.g. India cricket match shown on TV, or None
# - `prepared_qty`: portions the chef prepared (knows weekdays and festivals, NOT weather or cricket), or bottles chilled (beer, wine)
# - `sold_qty`: units sold
# - `wasted_qty`: units thrown away (unsold fresh items, or spillage and breakage for bottles)
# - `stockout`: Yes if customers wanted more than was prepared
#
# ⚠️ **Challenge:** there are hidden data problems in the output file. Do NOT read Section 5 of this
# notebook if you want to find them yourself. The answer key is in `SPOILER_answer_key.md`.

# %%
# ===== SECTION 1: SETUP =====
import numpy as np  # numbers and random draws
from pathlib import Path  # file paths that work on any computer
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"  # save into data/raw/
import pandas as pd  # tables of data (DataFrames)

SEED = 42  # fixed seed: anyone running this gets exactly the same data (reproducibility)
rng = np.random.default_rng(SEED)  # our random number generator, locked to the seed

START_DATE = "2025-01-01"  # first trading day
END_DATE = "2026-06-30"  # last trading day
CLOSED_DAYS = ["2025-12-25"]  # restaurant is closed on Christmas Day, so no rows

# %%
# ===== SECTION 2: THE MENU =====
# Each item: name, category, cuisine, price, unit cost, average daily sales, shelf life
MENU = [
    ("Butter Chicken", "Food", "Punjabi", 13.95, 4.20, 22, "Same day"),  # best seller
    ("Dal Makhani", "Food", "Punjabi", 10.50, 2.10, 18, "Same day"),  # slow cooked lentils
    ("Chole Bhature", "Food", "Punjabi", 11.00, 2.40, 14, "Same day"),  # chickpeas and fried bread
    ("Paneer Tikka", "Food", "Punjabi", 9.50, 3.10, 16, "Same day"),  # grilled paneer starter
    ("Dhokla", "Food", "Gujarati", 6.50, 1.20, 10, "Same day"),  # steamed gram flour snack
    ("Undhiyu", "Food", "Gujarati", 11.50, 2.80, 6, "Same day"),  # mixed winter vegetables
    ("Thepla with Chutney", "Food", "Gujarati", 6.00, 1.00, 9, "Same day"),  # spiced flatbread
    ("Khandvi", "Food", "Gujarati", 6.95, 1.30, 7, "Same day"),  # rolled gram flour snack
    ("Vada Pav", "Food", "Marathi", 5.50, 0.90, 20, "Same day"),  # Mumbai potato burger
    ("Misal Pav", "Food", "Marathi", 8.50, 1.60, 12, "Same day"),  # spicy sprouts curry
    ("Pav Bhaji", "Food", "Marathi", 9.00, 1.80, 17, "Same day"),  # mashed veg curry with buns
    ("Puran Poli", "Food", "Marathi", 6.50, 1.40, 6, "Same day"),  # sweet stuffed flatbread
    ("Masala Dosa", "Food", "South Indian", 9.95, 1.90, 21, "Same day"),  # crispy rice crepe
    ("Idli Sambar", "Food", "South Indian", 7.50, 1.20, 13, "Same day"),  # steamed rice cakes
    ("Medu Vada", "Food", "South Indian", 6.50, 1.10, 11, "Same day"),  # lentil doughnuts
    ("Chicken Chettinad", "Food", "South Indian", 13.50, 4.00, 12, "Same day"),  # pepper chicken curry
    ("Mango Lassi", "Drink", "Punjabi", 4.50, 1.10, 25, "Same day"),  # yoghurt mango drink
    ("Masala Chai", "Drink", "Punjabi", 3.00, 0.40, 30, "Same day"),  # spiced tea
    ("Chaas", "Drink", "Gujarati", 3.50, 0.50, 10, "Same day"),  # spiced buttermilk
    ("Sol Kadhi", "Drink", "Marathi", 4.00, 0.80, 6, "Same day"),  # kokum and coconut drink
    ("Filter Coffee", "Drink", "South Indian", 3.20, 0.45, 14, "Same day"),  # South Indian coffee
    ("Kingfisher Lager", "Drink", "Bar", 5.50, 1.60, 28, "Long"),  # bottled beer
    ("Cobra Beer", "Drink", "Bar", 5.50, 1.55, 24, "Long"),  # bottled beer
    ("House Red Wine 175ml", "Drink", "Bar", 6.95, 1.90, 9, "Long"),  # wine by the glass
]
menu_df = pd.DataFrame(MENU, columns=[
    "item_name", "category", "cuisine", "price_gbp", "unit_cost_gbp", "base_sales", "shelf_life"
])  # turn the list into a table

def price_tier(p):  # same three price groups as the Kaggle benchmark
    return "Low" if p < 6 else ("Moderate" if p < 10 else "High")
menu_df["price_tier"] = menu_df["price_gbp"].apply(price_tier)  # add the tier to each item

# Chef's over-prep "safety buffer" per price tier, tuned so waste matches the Kaggle benchmark
PREP_BUFFER = {"Low": -0.008, "Moderate": -0.005, "High": 0.050}  # 0.050 = 5% extra; a small minus = chef preps cheap items tight and cooks more if needed
BUSY_EXTRA_BUFFER = 0.015  # busy days (Fri/Sat/festivals): chef plays safe and preps a bit more
BENCHMARK_WASTE = {"Low": 0.055, "Moderate": 0.058, "High": 0.088, "Overall": 0.069}  # from Kaggle

# %%
# ===== SECTION 3: THE CALENDAR (dates, weather, festivals, events) =====
cal = pd.DataFrame({"date": pd.date_range(START_DATE, END_DATE, freq="D")})  # one row per day
cal = cal[~cal["date"].isin(pd.to_datetime(CLOSED_DAYS))].reset_index(drop=True)  # drop closed days
cal["day_of_week"] = cal["date"].dt.day_name()  # Monday, Tuesday...
month = cal["date"].dt.month  # month number 1 to 12

MONTH_TEMP = [5, 5, 7, 10, 13, 16, 18, 18, 15, 12, 8, 6]  # typical London average temp per month
cal["temperature_c"] = (np.array(MONTH_TEMP)[month - 1] + rng.normal(0, 3, len(cal))).round(1)  # add daily variation

def pick_weather(m):  # choose weather based on the season
    if m in (12, 1, 2):  # winter
        return rng.choice(["Sunny", "Cloudy", "Rainy", "Snow"], p=[0.15, 0.45, 0.35, 0.05])
    if m in (6, 7, 8):  # summer
        return rng.choice(["Sunny", "Cloudy", "Rainy", "Snow"], p=[0.45, 0.35, 0.20, 0.0])
    return rng.choice(["Sunny", "Cloudy", "Rainy", "Snow"], p=[0.25, 0.40, 0.35, 0.0])  # spring/autumn

cal["weather"] = [pick_weather(m) for m in month]  # one weather value per day

FESTIVALS = {  # festival name: list of (start, end) date ranges
    "Pongal": [("2025-01-14", "2025-01-15"), ("2026-01-14", "2026-01-15")],
    "Holi": [("2025-03-14", "2025-03-14"), ("2026-03-04", "2026-03-04")],
    "Vaisakhi": [("2025-04-13", "2025-04-14"), ("2026-04-14", "2026-04-14")],
    "Ganesh Chaturthi": [("2025-08-27", "2025-09-06")],
    "Navratri": [("2025-09-22", "2025-10-01")],
    "Diwali": [("2025-10-18", "2025-10-22")],
    "Christmas and New Year": [("2025-12-20", "2026-01-01")],
}
cal["festival"] = "None"  # default: no festival
for name, ranges in FESTIVALS.items():  # go through each festival
    for start, end in ranges:  # and each date range
        cal.loc[cal["date"].between(start, end), "festival"] = name  # label those days

cal["local_event"] = "None"  # default: no event
match_days = rng.choice(cal.index, size=24, replace=False)  # 24 random cricket match days
cal.loc[match_days, "local_event"] = "India cricket match"  # shown on the TV in the bar

# %%
# ===== SECTION 4: DEMAND, PREPARATION, SALES AND WASTE =====
df = cal.merge(menu_df, how="cross")  # every day x every menu item

DOW_MULT = {"Monday": 0.75, "Tuesday": 0.80, "Wednesday": 0.85, "Thursday": 0.95,
            "Friday": 1.25, "Saturday": 1.40, "Sunday": 1.15}  # busier at weekends
mult = df["day_of_week"].map(DOW_MULT).to_numpy()  # start with the day-of-week effect
# NOTE: effects are split in two groups.
#   KNOWN effects (day, trend, festivals): the chef plans for these.
#   HIDDEN effects (weather, temperature, cricket): the chef can't plan for these. The AI can learn them.

is_bar = (df["cuisine"] == "Bar").to_numpy()  # alcohol items
mult = mult * np.where(is_bar & df["day_of_week"].isin(["Friday", "Saturday"]), 1.25, 1.0)  # extra bar trade Fri/Sat

days_in = (df["date"] - pd.Timestamp(START_DATE)).dt.days.to_numpy()  # days since opening of data
mult = mult * (1 + 0.06 * days_in / 365)  # business grows about 6% per year

fest = df["festival"].to_numpy()  # festival for each row
cuis = df["cuisine"].to_numpy()  # cuisine for each row
meat = df["item_name"].isin(["Butter Chicken", "Chicken Chettinad"]).to_numpy()  # non-veg items
mult = mult * np.where(fest == "Diwali", 1.50, 1.0)  # Diwali: busy across the menu
mult = mult * np.where((fest == "Navratri") & (cuis == "Gujarati"), 1.80, 1.0)  # Navratri: Gujarati boost
mult = mult * np.where((fest == "Navratri") & meat, 0.70, 1.0)  # Navratri: many guests avoid meat
mult = mult * np.where((fest == "Ganesh Chaturthi") & (cuis == "Marathi"), 1.70, 1.0)  # Marathi boost
mult = mult * np.where((fest == "Pongal") & (cuis == "South Indian"), 1.60, 1.0)  # South Indian boost
mult = mult * np.where((fest == "Vaisakhi") & (cuis == "Punjabi"), 1.60, 1.0)  # Punjabi boost
mult = mult * np.where(fest == "Holi", 1.25, 1.0)  # Holi: busy day
mult = mult * np.where((fest == "Holi") & (df["item_name"] == "Mango Lassi").to_numpy(), 1.40, 1.0)  # lassi on Holi
mult = mult * np.where(fest == "Christmas and New Year", 1.20, 1.0)  # party season
mult = mult * np.where((fest == "Christmas and New Year") & is_bar, 1.30, 1.0)  # more drinking

known = mult.copy()  # everything so far is KNOWN to the chef

hidden = np.ones(len(df))  # HIDDEN effects start at 1 (no effect)
temp = df["temperature_c"].to_numpy()  # temperature for each row
cold_drinks = df["item_name"].isin(["Mango Lassi", "Chaas", "Sol Kadhi", "Kingfisher Lager", "Cobra Beer"]).to_numpy()
hot_drinks = df["item_name"].isin(["Masala Chai", "Filter Coffee"]).to_numpy()
curries = df["item_name"].isin(["Butter Chicken", "Dal Makhani", "Chicken Chettinad", "Undhiyu", "Misal Pav"]).to_numpy()
hidden = hidden * np.where(cold_drinks & (temp > 18), 1.35, 1.0)  # warm days: cold drinks sell more
hidden = hidden * np.where(hot_drinks & (temp > 18), 0.75, 1.0)  # warm days: fewer hot drinks
hidden = hidden * np.where(hot_drinks & (temp < 7), 1.30, 1.0)  # cold days: more chai and coffee
hidden = hidden * np.where(curries & (temp < 7), 1.15, 1.0)  # cold days: more warming curries
hidden = hidden * np.where(df["weather"].isin(["Rainy", "Snow"]).to_numpy(), 0.88, 1.0)  # bad weather: fewer diners

cricket = (df["local_event"] == "India cricket match").to_numpy()  # match on TV
snacks = df["item_name"].isin(["Vada Pav", "Medu Vada", "Paneer Tikka"]).to_numpy()  # match snacks
hidden = hidden * np.where(cricket & is_bar, 1.80, 1.0)  # beer flies during the match
hidden = hidden * np.where(cricket & snacks, 1.40, 1.0)  # snacks too

base = df["base_sales"].to_numpy()  # item's normal daily sales
expected = base * known * hidden  # TRUE average demand for that item that day
demand = np.maximum(np.round(expected * rng.normal(1, 0.08, len(df))), 0).astype(int)  # actual demand, about +/-8% day to day

chef_estimate = base * known  # what the chef THINKS demand will be (no weather or cricket knowledge)
busy = df["day_of_week"].isin(["Friday", "Saturday"]).to_numpy() | (df["festival"] != "None").to_numpy()  # busy days
buffer = df["price_tier"].map(PREP_BUFFER).to_numpy() + np.where(busy, BUSY_EXTRA_BUFFER, 0)  # chef's safety margin
fresh = (df["shelf_life"] == "Same day").to_numpy()  # freshly made items
prepared_fresh = np.round(chef_estimate * (1 + rng.normal(buffer, 0.04)))  # chef's prep: estimate + buffer + human variation
prepared_long = np.round(expected * rng.uniform(1.5, 2.5, len(df)))  # bottles chilled: generous stock
prepared = np.where(fresh, prepared_fresh, prepared_long).astype(int)  # pick the right one
prepared = np.maximum(prepared, 1)  # always prepare at least 1

sold = np.minimum(demand, prepared)  # you cannot sell more than you prepared
stockout = np.where(demand > prepared, "Yes", "No")  # customers wanted more than we had
spillage = rng.binomial(sold, 0.015)  # bottles: about 1.5% broken or spilled
wasted = np.where(fresh, prepared - sold, spillage)  # fresh: unsold is binned; bottles: only spillage

df["prepared_qty"] = prepared  # save into the table
df["sold_qty"] = sold
df["wasted_qty"] = wasted
df["stockout"] = stockout

df = df.sort_values(["date", "item_name"]).reset_index(drop=True)  # tidy order
df.insert(0, "record_id", np.arange(100001, 100001 + len(df)))  # unique ID per row
df = df[["record_id", "date", "day_of_week", "item_name", "category", "cuisine", "price_gbp",
         "price_tier", "unit_cost_gbp", "shelf_life", "weather", "temperature_c", "festival", "local_event",
         "prepared_qty", "sold_qty", "wasted_qty", "stockout"]]  # final column order

clean_rows = len(df)  # remember how many rows the clean version had
print(f"Clean dataset built: {clean_rows:,} rows")  # quick check

# %%
# ===== SECTION 4b: CALIBRATION CHECK (does our waste match the real Kaggle benchmark?) =====
fresh_df = df[df["shelf_life"] == "Same day"]  # waste benchmark only applies to freshly made items
ours = (fresh_df.groupby("price_tier")["wasted_qty"].sum() / fresh_df.groupby("price_tier")["prepared_qty"].sum())  # waste % per tier
ours["Overall"] = fresh_df["wasted_qty"].sum() / fresh_df["prepared_qty"].sum()  # overall waste %
for tier in ["Low", "Moderate", "High", "Overall"]:
    print(f"{tier:9s} ours {ours[tier]:.1%}   Kaggle benchmark {BENCHMARK_WASTE[tier]:.1%}")  # side by side
print(f"Stockout rate: {(df['stockout'] == 'Yes').mean():.1%}")  # how often we ran out

# %%
# ===== SECTION 5: ADD THE MESS (⚠️ SPOILERS! skip this cell if you want the challenge) =====
messy = df.copy().astype(object)  # object type lets us put text into number columns
messy["date"] = df["date"].dt.strftime("%Y-%m-%d")  # dates as text, like a real export
log = {}  # answer key: what we broke and how many rows

def pick(frac, mask=None):  # choose a random set of row numbers
    pool = messy.index if mask is None else messy.index[mask]
    return rng.choice(pool, size=int(len(messy) * frac), replace=False)

# 1. Mixed date formats
idx = pick(0.10)  # 10% as day/month/year
messy.loc[idx, "date"] = df.loc[idx, "date"].dt.strftime("%d/%m/%Y")
idx2 = pick(0.05)  # 5% as '4 Mar 2025'
messy.loc[idx2, "date"] = df.loc[idx2, "date"].dt.strftime("%d %b %Y")
log["1. Mixed date formats (YYYY-MM-DD, DD/MM/YYYY, 'DD Mon YYYY')"] = len(set(idx) | set(idx2))

# 2. Item name spelling, case and spaces
TYPOS = {"Butter Chicken": "Butter Chiken", "Masala Dosa": "Masala Dosaa", "Chole Bhature": "Chole Bature",
         "Pav Bhaji": "Paav Bhaji", "Idli Sambar": "Idly Sambar", "Dhokla": "Dhoklaa",
         "Kingfisher Lager": "Kingfisher", "Mango Lassi": "Mango lassi"}
idx = pick(0.05)
for i in idx:
    name = messy.at[i, "item_name"]
    style = rng.integers(0, 4)  # choose a type of mistake
    messy.at[i, "item_name"] = [name.lower(), name.upper(), f"  {name} ", TYPOS.get(name, name.lower())][style]
log["2. Item names with typos, wrong case or extra spaces"] = len(idx)

# 3. Category spelling
idx = pick(0.04)
for i in idx:
    opts = ["food", "FOOD", " Food"] if messy.at[i, "category"] == "Food" else ["drink", "Drinks", "DRINK"]
    messy.at[i, "category"] = rng.choice(opts)
log["3. Category written inconsistently (food, FOOD, Drinks...)"] = len(idx)

# 4. Prices stored as text with a pound sign
idx = pick(0.10)
messy.loc[idx, "price_gbp"] = [f"£{p:.2f}" for p in df.loc[idx, "price_gbp"]]
log["4. Price stored as text with a £ sign"] = len(idx)

# 5. Some temperatures recorded in Fahrenheit
idx = pick(0.03)
messy.loc[idx, "temperature_c"] = (df.loc[idx, "temperature_c"] * 9 / 5 + 32).round(1)
log["5. Temperature recorded in Fahrenheit instead of Celsius"] = len(idx)

# 6. Placeholder text instead of a temperature
idx = pick(0.01)
messy.loc[idx, "temperature_c"] = rng.choice(["unknown", "-", "N/A"], size=len(idx))
log["6. Placeholder text in temperature (unknown, -, N/A)"] = len(idx)

# 7. Missing values
idx_t = pick(0.04); messy.loc[idx_t, "temperature_c"] = np.nan  # temperature blanks
idx_w = pick(0.03); messy.loc[idx_w, "weather"] = np.nan  # weather blanks
idx_s = pick(0.015); messy.loc[idx_s, "sold_qty"] = np.nan  # sold quantity blanks
log["7. Missing values (temperature about 4%, weather about 3%, sold_qty about 1.5%)"] = len(idx_t) + len(idx_w) + len(idx_s)

# 8. Weather written inconsistently
idx = pick(0.06)
for i in idx:
    w = messy.at[i, "weather"]
    if isinstance(w, str):  # skip blanks
        messy.at[i, "weather"] = rng.choice([w.lower(), w.upper(), f"{w} ", "Rain" if w == "Rainy" else w.lower()])
log["8. Weather written inconsistently (rainy, RAINY, Rain, trailing spaces)"] = len(idx)

# 9. 'No festival' written many ways
none_rows = messy.index[messy["festival"] == "None"]
idx = rng.choice(none_rows, size=int(len(none_rows) * 0.3), replace=False)
messy.loc[idx, "festival"] = rng.choice([np.nan, "none", "No festival", "-"], size=len(idx))
log["9. 'No festival' written as blank, none, No festival or -"] = len(idx)

# 10. Negative quantities (typing errors)
idx_s = pick(0.005); messy.loc[idx_s, "sold_qty"] = -df.loc[idx_s, "sold_qty"]
idx_w = pick(0.003); messy.loc[idx_w, "wasted_qty"] = -df.loc[idx_w, "wasted_qty"]
log["10. Negative quantities in sold_qty or wasted_qty"] = len(idx_s) + len(idx_w)

# 11. Extreme outliers (extra zero typed)
idx = pick(0.003)
messy.loc[idx, "sold_qty"] = df.loc[idx, "sold_qty"] * 10
log["11. Outliers: sold_qty 10x too big (extra zero typed)"] = len(idx)

# 12. Waste does not add up (fresh items: should be prepared minus sold)
idx = pick(0.01, mask=(df["shelf_life"] == "Same day").to_numpy())
messy.loc[idx, "wasted_qty"] = df.loc[idx, "wasted_qty"] + rng.integers(3, 11, len(idx))
log["12. Logic error: wasted_qty does not equal prepared_qty minus sold_qty (fresh items)"] = len(idx)

# 13. Wrong day of week
idx = pick(0.005)
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
messy.loc[idx, "day_of_week"] = [rng.choice([d for d in days if d != cur]) for cur in df.loc[idx, "day_of_week"]]
log["13. day_of_week does not match the date"] = len(idx)

# 14. Stockout written inconsistently
idx = pick(0.02)
messy.loc[idx, "stockout"] = [rng.choice(["Y", "yes", "TRUE"]) if s == "Yes" else rng.choice(["N", "no", "FALSE"])
                              for s in df.loc[idx, "stockout"]]
log["14. Stockout written inconsistently (Y, yes, TRUE, N, no, FALSE)"] = len(idx)

# 15. Duplicate rows (till exported twice)
dups = messy.sample(frac=0.02, random_state=SEED)  # copy 2% of rows
messy = pd.concat([messy, dups])  # add them back in
log["15. Exact duplicate rows"] = len(dups)

# 16. Rows out of order
messy = messy.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle all rows
log["16. Rows are not in date order (shuffled)"] = len(messy)

# %%
# ===== SECTION 6: SAVE THE FILES =====
messy.to_csv(RAW_DIR / "uxbridge_lounge_sales_raw.csv", index=False)  # the messy file you will clean

with open(RAW_DIR.parent.parent / "SPOILER_answer_key.md", "w", encoding="utf-8") as f:  # answer key for checking your cleaning
    f.write("# Uxbridge Lounge raw data: answer key\n\n")
    f.write(f"Seed: {SEED}. Clean rows before mess: {clean_rows:,}. Rows in raw file: {len(messy):,}.\n\n")
    f.write("| Problem | Rows affected |\n|---|---|\n")
    for k, v in log.items():
        f.write(f"| {k} | {v:,} |\n")

print(f"Messy file saved: {len(messy):,} rows. Answer key saved (don't peek!).")  # done
