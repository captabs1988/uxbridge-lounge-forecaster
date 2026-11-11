# Uxbridge Lounge: Food and Drink Demand and Waste Forecaster

AI that tells an Indian restaurant's chef **how many portions of each dish to prepare tomorrow**,
so less food goes in the bin and fewer dishes sell out.
Level 7 AI Data Specialist capstone project (ST0763).

## Results at a glance (test period Mar to Jun 2026)
| | Chef today | With the AI (60% buffer) |
|---|---|---|
| Forecast error (MAE, portions) | 1.87 (same day last week) | **1.04** (44% better) |
| Fresh food waste cost | GBP 5,257 | **GBP 3,549 to 4,503** (14% to 33% less) |
| Stockout rate, fresh items | 23% | **13% to 29%** |
| 80% likely range, actual coverage | | 83% |
| Time to forecast a whole day | | 7 ms on a laptop CPU |

Ranges, not single numbers, because true demand on days the chef sold out is unknown.
Full details: [`docs/model_card.md`](docs/model_card.md).

## Project structure
```
data/raw/          original files, never edited (messy sales data + Kaggle benchmark)
data/processed/    clean data made by code
notebooks/         01 generate data, 02 cleaning (step by step), 03 EDA charts and tests
src/               the real code: config, generate_data, data_cleaning, features, train, predict
models/            saved models + metadata (versions, seed, settings)
api/               FastAPI web service
app/               Streamlit dashboard (Phase 9)
tests/             25 automated tests
docs/              business framing, lineage, governance, architecture, model card, metrics, figures
```

## How to run it
```bash
python -m venv .venv                      # create a private Python environment
.venv\Scripts\activate                    # Windows (Mac: source .venv/bin/activate)
pip install -r requirements.txt           # install libraries

python -m src.generate_data               # optional: rebuild the raw data (identical, seed 42)
python -m src.data_cleaning               # clean it: 17 steps, 18 checks
python -m src.train                       # train, evaluate, explain, save (about 1 minute)
python -m pytest                          # run the 25 tests
python -m src.predict                     # print a sample forecast

uvicorn api.main:app --reload             # start the API, then open http://127.0.0.1:8000/docs
```

## Methods in one paragraph
Daily sales for 24 menu items were generated (seed 42), with waste calibrated to a public Kaggle dataset,
then deliberately corrupted with 16 real-world problem types and cleaned by code (13,046 rows kept).
The model is gradient boosting (scikit-learn, Poisson loss), tuned with time-ordered cross validation and
compared against two baselines, linear regression and random forest. Demand on sold-out days is
"unconstrained" so the model learns true demand, not capped sales. Likely ranges use calibrated ratios
(split conformal). A second model estimates stockout risk for any planned prep quantity.
Explainability (permutation importance, SHAP), fairness across cuisines and price tiers, robustness and
a business simulation are all produced automatically by `src/train.py`.

## Data and privacy
Synthetic data for a fictional restaurant. No personal data. See [`docs/governance.md`](docs/governance.md).
