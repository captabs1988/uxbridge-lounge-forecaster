# Data Lineage (where every number comes from)

```
food_wastage_data.csv (Kaggle, public)          src/generate_data.py (seed 42)
   1,782 rows, 164 duplicates                         |
   used ONLY for waste ratios  ------------------>    | calibrates prep buffers by price tier
                                                      v
                                   data/raw/uxbridge_lounge_sales_raw.csv
                                   13,342 rows, 16 deliberate problem types
                                                      |
                                   src/data_cleaning.py (17 logged steps, 18 rules)
                                                      v
                                   data/processed/uxbridge_lounge_sales_clean.csv
                                   13,046 rows (97.8% kept)   + docs/cleaning_log.csv
                                                      |
                                   src/features.py (clues, time split)
                                                      v
                                   src/train.py -> models/*.joblib + docs/metrics.json + docs/model_card.md
                                                      |
                                   src/predict.py -> api/main.py (FastAPI) and app/ (Streamlit)
```

## Calibration check (Kaggle benchmark vs our data, fresh items)
| Price tier | Our waste | Kaggle waste |
|---|---|---|
| Low (under GBP 6) | 6.3% | 5.5% |
| Moderate (GBP 6 to 9.99) | 6.5% | 5.8% |
| High (GBP 10+) | 8.6% | 8.8% |
| **Overall** | **6.9%** | **6.9%** |

## Rules
- Raw files are never edited. Every change is made by code and logged.
- Rebuilding from scratch gives byte-identical files (`python -m src.generate_data` with seed 42).
- Values repaired from other records on the same day (temperature, weather) are traceable in the log.
