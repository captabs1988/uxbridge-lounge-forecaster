"""
train.py: Phases 5, 6 and 7 in one repeatable script.

Run from the project folder:
    python -m src.train

What it does, in order:
  Phase 5  Modelling   : baselines, 3 model types, tuning, prediction ranges,
                         stockout classifier, recommended prep quantity
  Phase 6  Evaluation  : accuracy, robustness, latency, cost, business simulation
  Phase 7  Governance  : explainability, fairness, model card, reproducibility info
Outputs go to models/ (saved models) and docs/ (metrics, charts, model card).

KEY IDEA: "censored demand"
On a stockout day, sold_qty is NOT the true demand. If the chef made 20 dosas and
sold all 20, customers may have wanted 26. Training on sold_qty alone teaches the
model to under-forecast. So we "unconstrain" demand on stockout days (estimate how
much was really wanted) before training. Retail forecasting teams do the same.
"""
import json  # save results as text files
import platform  # record the computer's Python version
import time  # measure speed

import joblib  # save and load trained models
import matplotlib

matplotlib.use("Agg")  # draw charts to files without opening windows
import matplotlib.pyplot as plt  # charts
import numpy as np  # maths
import pandas as pd  # tables
import sklearn  # machine learning library
from scipy.stats import norm  # the bell curve, used to estimate hidden demand
from sklearn.compose import ColumnTransformer  # applies different prep to different columns
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.inspection import permutation_importance  # explainability
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline  # chains prep + model into one object
from sklearn.preprocessing import OneHotEncoder  # turns words into 0/1 columns

from src import config
from src.data_cleaning import load_clean
from src.features import (CATEGORICAL, FEATURES, TARGET, add_baseline_lag, add_business_columns,
                          add_features, time_split)

np.random.seed(config.SEED)  # lock randomness


# =====================================================================
# Helpers
# =====================================================================
def make_pipeline(model):
    """Prep (one-hot the word columns, keep number columns) + the model, as one object."""
    prep = ColumnTransformer(
        [("words", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        remainder="passthrough")  # number columns go straight through
    return Pipeline([("prep", prep), ("model", model)])


def hgb(**kwargs):
    """Gradient boosting with our standard settings (fast, accurate, works on a laptop CPU)."""
    settings = dict(max_iter=400, learning_rate=0.05, random_state=config.SEED)
    settings.update(kwargs)
    return HistGradientBoostingRegressor(**settings)


def scores(actual, predicted):
    """The 3 standard forecast scores."""
    return {"MAE": round(mean_absolute_error(actual, predicted), 3),  # average miss, in portions
            "RMSE": round(float(np.sqrt(mean_squared_error(actual, predicted))), 3),  # punishes big misses
            "R2": round(r2_score(actual, predicted), 3)}  # 1.0 = perfect, 0 = no better than the average


def unconstrain_demand(train, n_rounds=5):
    """Estimate TRUE demand on stockout days (sold out = customers wanted at least that many).

    Each round: fit a model, look at how spread out its errors are on normal days,
    then for stockout days ask the bell curve: 'given demand was MORE than what we prepared,
    what's the expected demand?' Repeat so the estimate settles.
    """
    sold = train[TARGET].astype(float).to_numpy()
    prepared = train["prepared_qty"].to_numpy()
    censored = (train["stockout"] == "Yes").to_numpy()  # True = sold out, true demand unknown
    target = sold.copy()
    for _ in range(n_rounds):
        model = make_pipeline(hgb(loss="poisson")).fit(train[FEATURES], target)
        mean = model.predict(train[FEATURES])  # model's expected demand
        rel_spread = np.std((sold - mean)[~censored] / np.maximum(mean[~censored], 1))  # typical % miss on normal days
        spread = np.maximum(rel_spread * mean, 1.0)
        z = (prepared - mean) / spread  # how far above the expected demand the prep level was
        expected_if_above = mean + spread * norm.pdf(z) / np.clip(1 - norm.cdf(z), 1e-6, None)  # truncated bell curve
        target = np.where(censored, np.maximum(prepared, expected_if_above), sold)  # only stockout days change
    return target, float(rel_spread)


def simulate_prep(prep, rows):
    """Phase 6.6: what if the chef had prepped 'prep' instead? Fresh items only.

    Honest about the unknown: on chef stockout days we don't know true demand, so
    when our prep is HIGHER than the chef's we can't tell if we'd have sold out.
    We report a best case and a worst case.
    """
    chef_prep = rows["prepared_qty"].to_numpy()
    sold = rows[TARGET].to_numpy()
    censored = (rows["stockout"] == "Yes").to_numpy()
    cost = rows["unit_cost_gbp"].to_numpy()
    unknown = censored & (prep > chef_prep)  # can't tell what would have happened
    stockout_best = (prep < sold) | (censored & (prep <= chef_prep))  # definitely sold out
    stockout_worst = stockout_best | unknown
    waste_best = np.where(unknown, 0, np.maximum(prep - sold, 0))  # demand was high enough to sell it all
    waste_worst = np.where(unknown, prep - chef_prep, np.maximum(prep - sold, 0))  # demand was just above chef's prep
    return {
        "stockout_rate_best": round(float(stockout_best.mean()), 3),
        "stockout_rate_worst": round(float(stockout_worst.mean()), 3),
        "waste_cost_best_gbp": round(float((waste_best * cost).sum()), 2),
        "waste_cost_worst_gbp": round(float((waste_worst * cost).sum()), 2),
        "portions_prepared": int(prep.sum()),
    }


def save_bar(series, title, xlabel, filename, color="#2f6f8f"):
    """Simple horizontal bar chart saved to docs/figures/."""
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(series))))
    series.sort_values().plot.barh(ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=120)
    plt.close(fig)


# =====================================================================
# Main pipeline
# =====================================================================
def main():
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {}  # everything we measure ends up here

    # ---------- Phase 4: features + split ----------
    df = add_business_columns(add_baseline_lag(add_features(load_clean())))
    train, test = time_split(df)
    print(f"Train: {len(train):,} rows ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test : {len(test):,} rows ({test['date'].min().date()} to {test['date'].max().date()})")
    known = test["stockout"] == "No"  # on these test rows sold_qty IS the true demand
    metrics["data"] = {"train_rows": len(train), "test_rows": len(test),
                       "train_period": f"{train['date'].min().date()} to {train['date'].max().date()}",
                       "test_period": f"{test['date'].min().date()} to {test['date'].max().date()}",
                       "test_rows_with_true_demand_known": int(known.sum())}

    # ---------- Phase 5.0: unconstrain demand on stockout days ----------
    y_train, rel_spread = unconstrain_demand(train)
    metrics["unconstraining"] = {
        "stockout_rows_in_train": int((train["stockout"] == "Yes").sum()),
        "avg_demand_added_per_stockout_row": round(float((y_train - train[TARGET])[train["stockout"] == "Yes"].mean()), 2),
        "typical_relative_error": round(rel_spread, 3)}

    # ---------- Phase 5.1 to 5.4: baselines and models ----------
    # Scored on test days WITHOUT a stockout, because only there do we know true demand.
    experiments = {}
    has_lag = known & test["sold_last_week"].notna()
    experiments["Baseline: same day last week"] = scores(test.loc[has_lag, TARGET], test.loc[has_lag, "sold_last_week"])
    fresh_known = known & (test["shelf_life"] == "Same day")
    experiments["Baseline: chef's prep (fresh items)"] = scores(test.loc[fresh_known, TARGET], test.loc[fresh_known, "prepared_qty"])

    candidates = {
        "Linear Regression": (LinearRegression(), train[TARGET]),
        "Random Forest": (RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                                random_state=config.SEED), train[TARGET]),
        "Gradient Boosting": (hgb(), train[TARGET]),
        "Gradient Boosting (Poisson)": (hgb(loss="poisson"), train[TARGET]),
        "Gradient Boosting (Poisson + unconstrained demand)": (hgb(loss="poisson"), y_train),
    }
    train_times = {}
    for name, (model, y) in candidates.items():
        start = time.perf_counter()
        pipe = make_pipeline(model).fit(train[FEATURES], y)  # learn from the past
        train_times[name] = round(time.perf_counter() - start, 2)
        experiments[name] = scores(test.loc[known, TARGET], pipe.predict(test.loc[known, FEATURES]))
        experiments[name]["train_seconds"] = train_times[name]
        print(f"{name:52s} MAE {experiments[name]['MAE']:.3f}  ({train_times[name]}s)")

    # ---------- Phase 5.5: hyperparameter tuning (time-aware cross validation) ----------
    search = RandomizedSearchCV(
        make_pipeline(hgb(loss="poisson")),
        param_distributions={
            "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "model__max_iter": [200, 400, 600],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [10, 20, 40],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        },
        n_iter=12, cv=TimeSeriesSplit(n_splits=4),  # each fold trains on earlier months, tests on later
        scoring="neg_mean_absolute_error", random_state=config.SEED, n_jobs=-1)
    start = time.perf_counter()
    search.fit(train[FEATURES], y_train)
    tune_seconds = round(time.perf_counter() - start, 1)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    forecast_model = search.best_estimator_
    experiments["Tuned Gradient Boosting (final)"] = scores(test.loc[known, TARGET],
                                                            forecast_model.predict(test.loc[known, FEATURES]))
    print(f"Tuned in {tune_seconds}s, best settings: {best_params}")
    print(f"{'Tuned Gradient Boosting (final)':52s} MAE {experiments['Tuned Gradient Boosting (final)']['MAE']:.3f}")
    metrics["experiments"] = experiments
    metrics["tuning"] = {"method": "RandomizedSearchCV, 12 settings x 4 time-ordered folds",
                         "best_params": best_params, "seconds": tune_seconds}
    baseline_mae = experiments["Baseline: same day last week"]["MAE"]
    final_mae = experiments["Tuned Gradient Boosting (final)"]["MAE"]
    metrics["improvement_vs_baseline_pct"] = round(100 * (baseline_mae - final_mae) / baseline_mae, 1)

    # ---------- Phase 5.8 + 5.9: prediction ranges + recommended prep ----------
    # Method: CALIBRATED RATIOS (a simple form of "conformal prediction").
    # 1. Train a copy of the model on Jan to Dec 2025 only.
    # 2. Forecast Jan to Feb 2026 (data it hasn't seen) and record actual / forecast for every row.
    # 3. Those ratios tell us how far off forecasts usually are, e.g. the 10th percentile ratio
    #    is 0.88 means "on 10% of days demand was below 88% of the forecast".
    # Range = forecast x ratio(10%) to forecast x ratio(90%). Prep = forecast x ratio(service level).
    calib_fit = (train["date"] <= "2025-12-31").to_numpy()
    calib_model = make_pipeline(hgb(loss="poisson", **best_params)).fit(train.loc[calib_fit, FEATURES], y_train[calib_fit])
    ratios = y_train[~calib_fit] / np.maximum(calib_model.predict(train.loc[~calib_fit, FEATURES]), 0.5)
    levels = [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]  # 5%, 10% ... 95%
    ratio_table = {str(q): round(float(np.quantile(ratios, q)), 4) for q in levels}  # saved for the app
    r = lambda q: ratio_table[str(round(q, 2))]  # helper: ratio for a given level

    pred_test_all = forecast_model.predict(test[FEATURES])  # final model's forecast for every test row
    low, high = pred_test_all * r(config.INTERVAL_LOW), pred_test_all * r(config.INTERVAL_HIGH)
    inside = (test[TARGET] >= low) & (test[TARGET] <= high)  # did the real number land inside the range?
    metrics["prediction_interval"] = {
        "method": "calibrated ratios (split conformal), calibration window Jan to Feb 2026",
        "target_coverage": f"{int(100 * (config.INTERVAL_HIGH - config.INTERVAL_LOW))}%",
        "actual_coverage_on_known_days": round(float(inside[known].mean()), 3),
        "average_width_portions": round(float((high - low).mean()), 2),
        "ratios": {"low": r(config.INTERVAL_LOW), "high": r(config.INTERVAL_HIGH)}}

    # ---------- Phase 5.6: stockout classifier ----------
    # "If the chef preps THIS many, will we run out?" The strongest clue is how the planned prep
    # compares with the forecast, so we give the classifier that ratio directly.
    for part in (train, test):
        part["prep_vs_forecast"] = part["prepared_qty"] / np.maximum(forecast_model.predict(part[FEATURES]), 0.5)
    clf_features = FEATURES + ["prepared_qty", "prep_vs_forecast"]
    is_so = (train["stockout"] == "Yes").to_numpy()
    new_clf = lambda: make_pipeline(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                   random_state=config.SEED))
    # ALERT THRESHOLD: by default a classifier says "Yes" only above 50% risk. Stockouts are just
    # ~20% of rows, so that misses most of them. We pick a lower threshold on the calibration window
    # (NOT the test set, that would be cheating) by choosing the one with the best F1.
    calib_clf = new_clf().fit(train.loc[calib_fit, clf_features], is_so[calib_fit])
    calib_prob = calib_clf.predict_proba(train.loc[~calib_fit, clf_features])[:, 1]
    alert_threshold = float(max(np.round(np.arange(0.15, 0.55, 0.05), 2),
                                key=lambda t: f1_score(is_so[~calib_fit], calib_prob >= t)))
    stockout_model = new_clf().fit(train[clf_features], is_so)  # final classifier on all training data
    actual_so = test["stockout"] == "Yes"
    so_prob = stockout_model.predict_proba(test[clf_features])[:, 1]  # honest probabilities
    pred_so = so_prob >= alert_threshold  # warn above the chosen threshold
    tn, fp, fn, tp = confusion_matrix(actual_so, pred_so).ravel()
    metrics["stockout_classifier"] = {
        "accuracy": round(accuracy_score(actual_so, pred_so), 3),
        "precision": round(precision_score(actual_so, pred_so), 3),  # when it warns, how often it's right
        "recall": round(recall_score(actual_so, pred_so), 3),  # of real stockouts, how many it caught
        "f1": round(f1_score(actual_so, pred_so), 3),
        "confusion_matrix": {"true_no": int(tn), "false_alarm": int(fp), "missed": int(fn), "caught": int(tp)},
        "alert_threshold": alert_threshold,
        "average_predicted_risk": round(float(so_prob.mean()), 3), "actual_stockout_rate": round(float(actual_so.mean()), 3),
        "always_say_no_accuracy": round(float(1 - actual_so.mean()), 3)}  # the lazy baseline to beat

    # ---------- Phase 6.3: robustness ----------
    pred_test = forecast_model.predict(test[FEATURES])
    slices = {
        "All known days": known,
        "Weekend (Fri/Sat)": known & (test["is_weekend"] == 1),
        "Weekday": known & (test["is_weekend"] == 0),
        "Festival days": known & (test["is_festival"] == 1),
        "Cricket match days": known & (test["is_cricket"] == 1),
        "Rainy days": known & (test["weather"] == "Rainy"),
        "Warm days (>18C)": known & (test["temperature_c"] > 18),
    }
    metrics["robustness"] = {
        name: {"rows": int(mask.sum()),
               "MAE": round(mean_absolute_error(test.loc[mask, TARGET], pred_test[mask]), 3) if mask.sum() else None}
        for name, mask in slices.items()}
    noisy = test.loc[known, FEATURES].copy()
    noisy["temperature_c"] += np.random.default_rng(config.SEED).normal(0, 2, len(noisy))  # weather forecast off by ~2C
    metrics["robustness"]["Temperature forecast wrong by about 2C"] = {
        "rows": int(known.sum()),
        "MAE": round(mean_absolute_error(test.loc[known, TARGET], forecast_model.predict(noisy)), 3)}

    # ---------- Phase 6.4 + 6.5: latency and cost ----------
    one_day = test[test["date"] == test["date"].max()][FEATURES]  # 24 items = one day's forecast
    start = time.perf_counter()
    for _ in range(50):
        forecast_model.predict(one_day)
    latency_ms = 1000 * (time.perf_counter() - start) / 50
    total_train_s = sum(train_times.values()) + tune_seconds
    metrics["latency_and_cost"] = {
        "forecast_one_day_ms": round(latency_ms, 1),
        "total_training_seconds": round(total_train_s, 1),
        "estimated_energy_kwh": round(total_train_s * 65 / 3_600_000, 5),  # ASSUMPTION: laptop CPU at about 65 W
        "hardware": "Laptop CPU, no GPU needed",
        "running_cost": "GBP 0 (free tools, runs on CPU)"}

    # ---------- Phase 6.6: business simulation (fresh items, test period) ----------
    fresh = test[test["shelf_life"] == "Same day"].reset_index(drop=True)
    fresh_forecast = forecast_model.predict(fresh[FEATURES])
    chef = {"stockout_rate": round(float((fresh["stockout"] == "Yes").mean()), 3),
            "waste_cost_gbp": round(float(fresh["waste_cost_gbp"].sum()), 2),
            "portions_prepared": int(fresh["prepared_qty"].sum())}
    # The buffer ("service level") is a BUSINESS choice: more buffer = fewer stockouts but more waste.
    # So we show the manager the whole trade-off, then use the default from config.
    tradeoff = {str(q): simulate_prep(np.ceil(fresh_forecast * r(q)), fresh) for q in [0.5, 0.6, 0.7, 0.8, 0.9]}
    ai = tradeoff[str(config.SERVICE_LEVEL)]
    mid_waste = (ai["waste_cost_best_gbp"] + ai["waste_cost_worst_gbp"]) / 2
    metrics["business_simulation"] = {
        "period": metrics["data"]["test_period"], "items": "fresh items only (bottles keep, so no waste)",
        "service_level": config.SERVICE_LEVEL, "chef_current_practice": chef, "ai_recommended_prep": ai,
        "waste_saving_range_pct": [round(100 * (1 - ai["waste_cost_worst_gbp"] / chef["waste_cost_gbp"]), 1),
                                   round(100 * (1 - ai["waste_cost_best_gbp"] / chef["waste_cost_gbp"]), 1)],
        "waste_saving_mid_gbp": round(chef["waste_cost_gbp"] - mid_waste, 2),
        "tradeoff_by_service_level": tradeoff}
    fig, ax = plt.subplots(figsize=(7, 4.5))  # trade-off chart: waste vs stockouts
    for q, v in tradeoff.items():
        mid_w = (v["waste_cost_best_gbp"] + v["waste_cost_worst_gbp"]) / 2
        mid_s = 100 * (v["stockout_rate_best"] + v["stockout_rate_worst"]) / 2
        ax.scatter(mid_w, mid_s, color="#2f6f8f", s=60)
        ax.annotate(f"AI {int(float(q) * 100)}%", (mid_w, mid_s), textcoords="offset points", xytext=(6, 4))
    ax.scatter(chef["waste_cost_gbp"], 100 * chef["stockout_rate"], color="#c0392b", s=90, marker="X")
    ax.annotate("Chef today", (chef["waste_cost_gbp"], 100 * chef["stockout_rate"]),
                textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Waste cost, Mar to Jun 2026 (GBP, midpoint)")
    ax.set_ylabel("Stockout rate % (midpoint)")
    ax.set_title("Waste vs stockouts: chef today vs AI at different buffers")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "waste_vs_stockout_tradeoff.png", dpi=120)
    plt.close(fig)

    # ---------- Phase 7.2: fairness (does it work equally well for every group?) ----------
    known_rows = test[known].copy()
    known_rows["pred"] = pred_test[known.to_numpy()]
    known_rows["abs_err"] = (known_rows["pred"] - known_rows[TARGET]).abs()
    overall_wape = known_rows["abs_err"].sum() / known_rows[TARGET].sum()
    fairness = {}
    for group in ["category", "cuisine", "price_tier"]:
        g = known_rows.groupby(group).agg(abs_err=("abs_err", "sum"), actual=(TARGET, "sum"), rows=(TARGET, "size"))
        g["WAPE"] = g["abs_err"] / g["actual"]  # % error, fair across busy and quiet items
        g["ratio_to_overall"] = g["WAPE"] / overall_wape
        fairness[group] = {k: {"rows": int(v["rows"]), "WAPE": round(v["WAPE"], 3),
                               "ratio_to_overall": round(v["ratio_to_overall"], 2),
                               "flag": bool(v["ratio_to_overall"] > 1.25)} for k, v in g.iterrows()}
    metrics["fairness"] = {"overall_WAPE": round(overall_wape, 3),
                           "rule": "flag a group if its error is more than 1.25x the overall error",
                           "groups": fairness}
    by_cuisine = pd.Series({k: v["WAPE"] for k, v in fairness["cuisine"].items()})
    save_bar(100 * by_cuisine, "Forecast error by cuisine (lower is better)", "WAPE %", "fairness_by_cuisine.png")

    # ---------- Phase 7.1: explainability ----------
    sample = test[known].sample(n=min(2000, int(known.sum())), random_state=config.SEED)
    perm = permutation_importance(forecast_model, sample[FEATURES], sample[TARGET], n_repeats=5,
                                  random_state=config.SEED, scoring="neg_mean_absolute_error")
    importance = pd.Series(perm.importances_mean, index=FEATURES).round(3)
    metrics["feature_importance"] = importance.sort_values(ascending=False).to_dict()
    save_bar(importance, "What drives the forecast? (permutation importance)",
             "Increase in MAE when this clue is scrambled", "feature_importance.png")
    try:  # SHAP: per-prediction explanation (optional library)
        import shap
        prep_step = forecast_model.named_steps["prep"]
        x_sample = pd.DataFrame(prep_step.transform(sample[FEATURES][:500]), columns=prep_step.get_feature_names_out())
        shap_values = shap.TreeExplainer(forecast_model.named_steps["model"]).shap_values(x_sample)
        shap.summary_plot(shap_values, x_sample, show=False, max_display=15)
        plt.title("SHAP: how each clue pushes the forecast up or down")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / "shap_summary.png", dpi=120)
        plt.close("all")
        metrics["shap"] = "saved docs/figures/shap_summary.png"
    except Exception as err:  # never crash the pipeline over an optional chart
        metrics["shap"] = f"skipped: {err}"

    # ---------- Model comparison chart ----------
    exp_mae = pd.Series({k: v["MAE"] for k, v in experiments.items() if "chef" not in k})
    save_bar(exp_mae, "Model comparison on test period (lower is better)", "MAE (portions)", "model_comparison.png")

    # ---------- Phase 7.8: save models + reproducibility info ----------
    menu = (df.drop_duplicates("item_name")[["item_name", "category", "cuisine", "price_gbp", "price_tier",
                                             "unit_cost_gbp", "shelf_life"]]
            .set_index("item_name").loc[config.MENU].reset_index())  # official menu order
    joblib.dump(forecast_model, config.FORECAST_MODEL_FILE)
    joblib.dump(stockout_model, config.STOCKOUT_MODEL_FILE)
    metadata = {
        "project": "Uxbridge Lounge food and drink demand forecaster",
        "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "seed": config.SEED, "features": FEATURES, "target": TARGET,
        "train_end": config.TRAIN_END, "test_start": config.TEST_START,
        "service_level": config.SERVICE_LEVEL,
        "interval": [config.INTERVAL_LOW, config.INTERVAL_HIGH],
        "ratio_table": ratio_table,
        "stockout_alert_threshold": alert_threshold,  # used by predict.py for ranges and prep at ANY service level
        "best_params": best_params, "menu": menu.to_dict(orient="records"),
        "test_MAE": final_mae,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit-learn": sklearn.__version__},
    }
    config.METADATA_FILE.write_text(json.dumps(metadata, indent=2, default=str))
    config.METRICS_FILE.write_text(json.dumps(metrics, indent=2, default=str))
    write_model_card(metrics, metadata)
    print(f"\nFinal MAE {final_mae} portions, {metrics['improvement_vs_baseline_pct']}% better than last-week baseline")
    print(f"Waste saving {metrics['business_simulation']['waste_saving_range_pct']} % (worst to best case)")
    print("Saved models/ and docs/ (metrics.json, model_card.md, figures/)")


def write_model_card(m, meta):
    """Phase 7.7: the model card, the 'nutrition label' of the model."""
    exp = m["experiments"]
    sim = m["business_simulation"]
    fair_flags = [f"{grp}: {name}" for grp, vals in m["fairness"]["groups"].items()
                  for name, v in vals.items() if v["flag"]]
    rows = "\n".join(f"| {k} | {v['MAE']} | {v['RMSE']} | {v['R2']} |" for k, v in exp.items())
    top = list(m["feature_importance"].items())[:5]
    card = f"""# Model Card: Uxbridge Lounge Demand Forecaster

_Generated automatically by `src/train.py` on {meta['trained_at']}._

## 1. What it does
Forecasts how many portions of each of the 24 menu items Uxbridge Lounge will sell on a given day,
gives an 80% likely range, and recommends how many to prepare. A second model estimates the risk of
running out for a planned prep quantity.

**Intended users:** head chef, bar manager, restaurant manager. **Decision support only:** the chef keeps the final say.
**Not for:** staffing, pricing, or any decision about individual people.

## 2. Data
- Source: synthetic daily sales for a fictional restaurant (Jan 2025 to Jun 2026), seed {meta['seed']}.
  Waste levels calibrated to the public Kaggle food wastage dataset (overall waste about 6.9%).
- Cleaning: 16 problem types fixed, 18 validation rules passed (see `docs/cleaning_log.csv`).
- Train: {m['data']['train_period']} ({m['data']['train_rows']:,} rows). Test: {m['data']['test_period']} ({m['data']['test_rows']:,} rows).
- Personal data: **none**. No customer, staff or payment information.

## 3. Model
- Final model: Gradient Boosting (scikit-learn HistGradientBoostingRegressor), Poisson loss (suits counts).
- Tuned with time-ordered cross validation. Best settings: `{meta['best_params']}`.
- Demand on stockout days was "unconstrained" ({m['unconstraining']['stockout_rows_in_train']:,} rows,
  about +{m['unconstraining']['avg_demand_added_per_stockout_row']} portions each), so the model learns true demand, not capped sales.

## 4. Performance (test period, days with no stockout so true demand is known)
| Model | MAE | RMSE | R2 |
|---|---|---|---|
{rows}

- **{m['improvement_vs_baseline_pct']}% lower error than the "same day last week" baseline.**
- 80% prediction range (calibrated ratios): actual coverage {m['prediction_interval']['actual_coverage_on_known_days']:.0%}, average width {m['prediction_interval']['average_width_portions']} portions.
- Stockout classifier (alerts above {m['stockout_classifier']['alert_threshold']:.0%} risk, threshold chosen on the calibration window): precision {m['stockout_classifier']['precision']}, recall {m['stockout_classifier']['recall']}, F1 {m['stockout_classifier']['f1']}.
- Speed: one day's forecast in {m['latency_and_cost']['forecast_one_day_ms']} ms on a laptop CPU.

## 5. Business impact (simulation, fresh items, test period)
- Chef's actual waste: GBP {sim['chef_current_practice']['waste_cost_gbp']:,.2f}, stockout rate {sim['chef_current_practice']['stockout_rate']:.0%}.
- AI recommended prep waste: GBP {sim['ai_recommended_prep']['waste_cost_best_gbp']:,.2f} to {sim['ai_recommended_prep']['waste_cost_worst_gbp']:,.2f}
  (**{sim['waste_saving_range_pct'][0]}% to {sim['waste_saving_range_pct'][1]}% less**), stockout rate
  {sim['ai_recommended_prep']['stockout_rate_best']:.0%} to {sim['ai_recommended_prep']['stockout_rate_worst']:.0%}.
- It's a range because true demand on the chef's stockout days is unknown.

## 6. Explainability
Top drivers (permutation importance): {', '.join(f'{k} ({v})' for k, v in top)}.
Charts: `docs/figures/feature_importance.png`, `docs/figures/shap_summary.png`.

## 7. Fairness
Error compared across food vs drink, the cuisines, and price tiers (WAPE). Rule: flag if more than 1.25x the overall error.
Flagged groups: {', '.join(fair_flags) if fair_flags else 'none'}. See `docs/metrics.json`.

## 8. Limitations and risks
- Trained on synthetic data: the model partly learns the rules used to generate it. Real POS data is needed before live use.
- Only one year of festival history; Diwali, Navratri and Ganesh Chaturthi are not in the test period.
- Weather inputs are forecasts in real life; error rises if the forecast is wrong (see robustness in `metrics.json`).
- Allergens and food safety are NOT handled by this model and must follow existing procedures (Natasha's Law).

## 9. Reproducibility
Seed {meta['seed']} everywhere. Rebuild everything with:
`python -m src.generate_data`, `python -m src.data_cleaning`, `python -m src.train`.
Versions: {meta['versions']}.
"""
    config.MODEL_CARD_FILE.write_text(card, encoding="utf-8")


if __name__ == "__main__":
    main()
