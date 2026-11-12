"""
streamlit_app.py: Phase 9, the chef's dashboard (the "front of house").

Run it from the project folder:
    streamlit run app/streamlit_app.py
Then open http://localhost:8501

Design decision (good to explain to an assessor):
The app calls src.predict.forecast_day() DIRECTLY, the same function the API uses.
One brain (predict.py), two doors (this app for people, the API for other systems).
So the app and the API can never disagree, and the app still works if the API is down.
"""
import json  # read the metrics file
import sys  # lets us tell Python where the project folder is
from datetime import date, timedelta  # dates for the date picker
from pathlib import Path  # file paths that work on Windows and Mac

ROOT = Path(__file__).resolve().parent.parent  # the project's top folder (one level above app/)
if str(ROOT) not in sys.path:  # if Python can't see the project folder yet...
    sys.path.insert(0, str(ROOT))  # ...add it, so "from src import ..." works on Streamlit Cloud too

import pandas as pd  # tables
import streamlit as st  # the web app library

from src import config  # all settings in one place
from src.predict import forecast_day, load_artifacts, summary  # the ONE forecast function

# ---------- Page setup (must be the first Streamlit command) ----------
st.set_page_config(page_title="Uxbridge Lounge Forecaster", page_icon="🍛", layout="wide")


# ---------- Cached helpers (run once, then reuse = fast app) ----------
@st.cache_data  # remember the result so the file is only read once
def load_metrics():
    """All evaluation results saved by src/train.py."""
    return json.loads(config.METRICS_FILE.read_text())


@st.cache_data  # same inputs = same answer, so reuse it instead of re-running the model
def get_forecast(day, weather, temp, festival, event, service_level, plan_items=None):
    """Wrap forecast_day so Streamlit can cache it. plan_items is a tuple because caches need fixed values."""
    plan = dict(plan_items) if plan_items else None  # turn the tuple back into {item: portions}
    return forecast_day(day, weather, temp, festival, event, plan, service_level)


meta = load_artifacts()["meta"]  # model info: trained date, menu, settings
metrics = load_metrics()  # test results for the "Model and governance" tab
menu = pd.DataFrame(meta["menu"])  # the 24 items with price, cost and shelf life

# ---------- Sidebar: the chef's inputs ----------
st.sidebar.title("🍛 Uxbridge Lounge")  # restaurant name at the top of the sidebar
st.sidebar.caption("Plan tomorrow's prep in 30 seconds")  # one-line promise to the user

day = st.sidebar.date_input("📅 Day to forecast", value=date.today() + timedelta(days=1))  # default = tomorrow
weather = st.sidebar.selectbox("🌦️ Weather forecast", config.WEATHER_TYPES)  # only values the model knows
temp = st.sidebar.slider("🌡️ Temperature (°C)", -10, 35, 15)  # same safe range the API enforces
festival = st.sidebar.selectbox("🪔 Festival", config.FESTIVALS)  # e.g. Diwali, Holi
event = st.sidebar.selectbox("🏏 Local event", config.LOCAL_EVENTS)  # e.g. India cricket match

st.sidebar.divider()  # thin line to separate business settings
service_pct = st.sidebar.slider(  # the buffer for fresh food, as a % (easier for chefs than 0.6)
    "🛡️ Safety buffer for fresh food", 50, 90, int(config.SERVICE_LEVEL * 100), step=5,
    help="Higher = fewer stockouts but more waste. 60% was the best balance in testing.")
service_level = service_pct / 100  # back to 0.6 style for the model

st.sidebar.divider()
st.sidebar.caption(f"Model trained: {meta['trained_at'][:10]} · seed {meta['seed']}")  # reproducibility at a glance
st.sidebar.caption("Decision support only. The chef has the final say.")  # human in the loop

# ---------- Run the forecast ----------
table = get_forecast(day, weather, temp, festival, event, service_level)  # 24 rows, one per menu item
totals = summary(table)  # header numbers: portions, revenue, waste, risky items

# ---------- Header and KPI cards ----------
st.title("Food and Drink Demand Forecaster")  # page title
st.caption(f"{day:%A %d %B %Y} · {weather}, {temp}°C"
           + (f" · {festival}" if festival != "None" else "")
           + (f" · {event}" if event != "None" else ""))  # the day's conditions in one line

k1, k2, k3, k4 = st.columns(4)  # four cards side by side
k1.metric("🍽️ Expected portions", f"{totals['total_forecast_portions']:.0f}")  # what we think will sell
k2.metric("👨‍🍳 Recommended prep", f"{totals['total_recommended_prep']}")  # what to make
k3.metric("💷 Expected revenue", f"£{totals['expected_revenue_gbp']:,.0f}")  # money in
k4.metric("🗑️ Expected waste cost", f"£{totals['expected_waste_cost_gbp']:,.2f}")  # money in the bin

if totals["high_risk_items"]:  # any item likely to run out even with our prep?
    st.warning("⚠️ Stockout risk above 30%: " + ", ".join(totals["high_risk_items"])
               + ". Consider a few extra portions or a backup dish.")  # tell the chef plainly
else:
    st.success("✅ No high stockout risks for this day.")  # all clear

# ---------- Tabs ----------
tab_prep, tab_chef, tab_model, tab_about = st.tabs(
    ["📋 Prep sheet", "⚖️ Chef vs AI", "🔍 Model and governance", "ℹ️ About"])  # four screens

# Columns shown to the chef, with friendly names
SHOW = {"item_name": "Item", "cuisine": "Cuisine", "forecast": "Forecast",
        "likely_low": "Low (80%)", "likely_high": "High (80%)", "recommended_prep": "Prep this many",
        "stockout_risk_pct": "Stockout risk %", "expected_waste_cost_gbp": "Waste £"}

with tab_prep:  # TAB 1: what to make today
    food = table[table["category"] == "Food"]  # kitchen items
    drink = table[table["category"] == "Drink"]  # bar and soft drinks
    st.subheader("🍛 Kitchen")
    st.dataframe(food[list(SHOW)].rename(columns=SHOW), hide_index=True, width="stretch",
                 column_config={"Stockout risk %": st.column_config.ProgressColumn(  # risk as a bar
                     min_value=0, max_value=100, format="%d%%")})
    st.subheader("🍺 Bar and drinks")
    st.dataframe(drink[list(SHOW)].rename(columns=SHOW), hide_index=True, width="stretch",
                 column_config={"Stockout risk %": st.column_config.ProgressColumn(
                     min_value=0, max_value=100, format="%d%%")})
    st.caption("Forecast = most likely sales. Low/High = sales land in this range on about 8 days out of 10. "
               "Bottles use a bigger buffer (90%) because they keep, so they never count as waste.")
    st.download_button("⬇️ Download prep sheet (CSV)",  # the chef can print it or send it to the team
                       table.to_csv(index=False).encode("utf-8"),
                       file_name=f"prep_sheet_{day}.csv", mime="text/csv")

with tab_chef:  # TAB 2: compare the chef's own plan with the AI
    st.subheader("Type your own prep plan and compare")
    st.caption("Change the 'Your plan' numbers. The AI shows the stockout risk and waste of YOUR plan next to its own.")
    editor = st.data_editor(  # an editable table, like a mini spreadsheet
        table[["item_name", "recommended_prep"]].assign(your_plan=table["recommended_prep"]),
        column_config={"item_name": "Item", "recommended_prep": "AI prep",
                       "your_plan": st.column_config.NumberColumn("Your plan", min_value=0, max_value=500, step=1)},
        disabled=["item_name", "recommended_prep"],  # only "Your plan" can be edited
        hide_index=True, width="stretch", key="chef_plan")
    plan = tuple(zip(editor["item_name"], editor["your_plan"].astype(int)))  # {item: portions} as a tuple
    compared = get_forecast(day, weather, temp, festival, event, service_level, plan)  # re-run with the chef's plan
    fresh = compared["item_name"].map(menu.set_index("item_name")["shelf_life"]).eq("Same day")  # bin if unsold
    cost = compared["item_name"].map(menu.set_index("item_name")["unit_cost_gbp"])  # cost per portion
    def waste(prep):  # expected waste £ = portions left over x cost, fresh items only (same rule for both plans)
        return ((prep - compared["forecast"]).clip(lower=0) * cost)[fresh].sum()
    chef_waste = waste(compared["chef_planned_prep"])  # waste of YOUR plan
    ai_waste = waste(compared["recommended_prep"])  # waste of the AI plan
    c1, c2, c3 = st.columns(3)
    c1.metric("Your expected waste", f"£{chef_waste:,.2f}",
              delta=f"£{chef_waste - ai_waste:,.2f} vs AI", delta_color="inverse")  # red if you waste more
    c2.metric("Your high-risk items", int((compared["chef_stockout_risk_pct"] >= 30).sum()),
              delta=int((compared["chef_stockout_risk_pct"] >= 30).sum() - compared["stockout_alert"].sum()),
              delta_color="inverse")  # red if you have more risky items
    c3.metric("Your total prep", int(compared["chef_planned_prep"].sum()),
              delta=int(compared["chef_planned_prep"].sum() - compared["recommended_prep"].sum()))
    changed = compared[compared["chef_planned_prep"] != compared["recommended_prep"]]  # only rows the chef changed
    if not changed.empty:
        st.dataframe(changed[["item_name", "forecast", "recommended_prep", "stockout_risk_pct",
                              "chef_planned_prep", "chef_stockout_risk_pct"]].rename(columns={
            "item_name": "Item", "forecast": "Forecast", "recommended_prep": "AI prep",
            "stockout_risk_pct": "AI risk %", "chef_planned_prep": "Your prep", "chef_stockout_risk_pct": "Your risk %"}),
            hide_index=True, width="stretch")

with tab_model:  # TAB 3: evidence for the assessor (evaluation, explainability, fairness)
    exp = pd.DataFrame(metrics["experiments"]).T.reset_index().rename(columns={"index": "Model"})  # one row per model
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test MAE", meta["test_MAE"], help="Average miss in portions per item per day (lower = better)")
    m2.metric("vs baseline", f"{metrics['improvement_vs_baseline_pct']}% better",
              help="Baseline = same day last week")
    m3.metric("80% range coverage", f"{metrics['prediction_interval']['actual_coverage_on_known_days']:.0%}")
    m4.metric("Forecast speed", f"{metrics['latency_and_cost']['forecast_one_day_ms']} ms")
    st.subheader("📊 Experiments (test period Mar to Jun 2026)")
    st.dataframe(exp[["Model", "MAE", "RMSE", "R2"]], hide_index=True, width="stretch")
    clf = metrics["stockout_classifier"]  # the yes/no "will we run out?" model
    st.caption(f"Stockout alert model: precision {clf['precision']:.2f}, recall {clf['recall']:.2f}, "
               f"F1 {clf['f1']:.2f} at a {clf['alert_threshold']:.0%} threshold.")
    st.subheader("🧠 Why the model predicts what it does")
    f1, f2 = st.columns(2)
    f1.image(str(config.FIGURES_DIR / "shap_summary.png"), caption="SHAP: which clues push the forecast up or down")
    f2.image(str(config.FIGURES_DIR / "feature_importance.png"), caption="Permutation importance")
    st.subheader("⚖️ Fairness and trade-offs")
    g1, g2 = st.columns(2)
    g1.image(str(config.FIGURES_DIR / "fairness_by_cuisine.png"), caption="Error by cuisine: no group flagged")
    g2.image(str(config.FIGURES_DIR / "waste_vs_stockout_tradeoff.png"), caption="Waste vs stockouts by buffer")
    with st.expander("📄 Full model card"):  # the governance document, folded away to keep the page short
        st.markdown(config.MODEL_CARD_FILE.read_text())

with tab_about:  # TAB 4: plain-English summary for non-technical users
    st.markdown("""
**What it does:** forecasts how many portions of each of the 24 dishes and drinks will sell on a day,
recommends how many to prepare, and warns about likely stockouts.

**Who it's for:** head chef, bar manager, restaurant manager. **The chef keeps the final say.**

**Data:** synthetic sales for a fictional restaurant (Jan 2025 to Jun 2026), seed 42.
No customer, staff or payment data (no personal data under UK GDPR).

**Limitations:** trained on synthetic data, so real till data is needed before live use.
Festival days have about 2x the normal error (little history). Allergens are not handled by this tool.

**Also available as an API** for other systems (till, ordering): see the project README.
""")
