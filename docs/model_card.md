# Model Card: Uxbridge Lounge Demand Forecaster

_Generated automatically by `src/train.py` on 2026-11-11T18:19:41._

## 1. What it does
Forecasts how many portions of each of the 24 menu items Uxbridge Lounge will sell on a given day,
gives an 80% likely range, and recommends how many to prepare. A second model estimates the risk of
running out for a planned prep quantity.

**Intended users:** head chef, bar manager, restaurant manager. **Decision support only:** the chef keeps the final say.
**Not for:** staffing, pricing, or any decision about individual people.

## 2. Data
- Source: synthetic daily sales for a fictional restaurant (Jan 2025 to Jun 2026), seed 42.
  Waste levels calibrated to the public Kaggle food wastage dataset (overall waste about 6.9%).
- Cleaning: 16 problem types fixed, 18 validation rules passed (see `docs/cleaning_log.csv`).
- Train: 2025-01-01 to 2026-02-28 (10,125 rows). Test: 2026-03-01 to 2026-06-30 (2,921 rows).
- Personal data: **none**. No customer, staff or payment information.

## 3. Model
- Final model: Gradient Boosting (scikit-learn HistGradientBoostingRegressor), Poisson loss (suits counts).
- Tuned with time-ordered cross validation. Best settings: `{'min_samples_leaf': 10, 'max_leaf_nodes': 15, 'max_iter': 600, 'learning_rate': 0.08, 'l2_regularization': 0.1}`.
- Demand on stockout days was "unconstrained" (2,324 rows,
  about +1.59 portions each), so the model learns true demand, not capped sales.

## 4. Performance (test period, days with no stockout so true demand is known)
| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Baseline: same day last week | 1.866 | 3.762 | 0.853 |
| Baseline: chef's prep (fresh items) | 1.455 | 2.201 | 0.909 |
| Linear Regression | 2.249 | 3.777 | 0.852 |
| Random Forest | 1.129 | 1.953 | 0.961 |
| Gradient Boosting | 1.099 | 1.723 | 0.969 |
| Gradient Boosting (Poisson) | 1.066 | 1.607 | 0.973 |
| Gradient Boosting (Poisson + unconstrained demand) | 1.051 | 1.632 | 0.972 |
| Tuned Gradient Boosting (final) | 1.039 | 1.582 | 0.974 |

- **44.3% lower error than the "same day last week" baseline.**
- 80% prediction range (calibrated ratios): actual coverage 83%, average width 3.75 portions.
- Stockout classifier (alerts above 30% risk, threshold chosen on the calibration window): precision 0.586, recall 0.552, F1 0.568.
- Speed: one day's forecast in 16.0 ms on a laptop CPU.

## 5. Business impact (simulation, fresh items, test period)
- Chef's actual waste: GBP 5,257.05, stockout rate 23%.
- AI recommended prep waste: GBP 3,548.55 to 4,503.25
  (**14.3% to 32.5% less**), stockout rate
  13% to 29%.
- It's a range because true demand on the chef's stockout days is unknown.

## 6. Explainability
Top drivers (permutation importance): item_name (4.622), cuisine (3.041), is_weekend (1.86), day_of_week (1.516), category (1.192).
Charts: `docs/figures/feature_importance.png`, `docs/figures/shap_summary.png`.

## 7. Fairness
Error compared across food vs drink, the cuisines, and price tiers (WAPE). Rule: flag if more than 1.25x the overall error.
Flagged groups: none. See `docs/metrics.json`.

## 8. Limitations and risks
- Trained on synthetic data: the model partly learns the rules used to generate it. Real POS data is needed before live use.
- Only one year of festival history; Diwali, Navratri and Ganesh Chaturthi are not in the test period.
- Weather inputs are forecasts in real life; error rises if the forecast is wrong (see robustness in `metrics.json`).
- Allergens and food safety are NOT handled by this model and must follow existing procedures (Natasha's Law).

## 9. Reproducibility
Seed 42 everywhere. Rebuild everything with:
`python -m src.generate_data`, `python -m src.data_cleaning`, `python -m src.train`.
Versions: {'python': '3.12.10', 'pandas': '3.1.1', 'numpy': '2.5.4', 'scikit-learn': '1.9.1'}.
