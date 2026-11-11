"""
config.py: ONE place for every setting in the project.

Why: if a path, date or seed changes, you change it here once,
not in ten different files. Every other file imports from here.
"""
from pathlib import Path  # Path makes file paths work on Windows AND Mac

# ---------- Folders ----------
ROOT = Path(__file__).resolve().parent.parent  # the project's top folder (one level above src/)
RAW_DIR = ROOT / "data" / "raw"  # original files, NEVER edited
PROCESSED_DIR = ROOT / "data" / "processed"  # files created by our code
MODELS_DIR = ROOT / "models"  # saved, trained models
DOCS_DIR = ROOT / "docs"  # reports, model card, cleaning log
FIGURES_DIR = DOCS_DIR / "figures"  # saved charts

# ---------- Files ----------
RAW_SALES_FILE = RAW_DIR / "uxbridge_lounge_sales_raw.csv"  # messy restaurant data
KAGGLE_FILE = RAW_DIR / "food_wastage_data.csv"  # public benchmark used to calibrate waste
CLEAN_SALES_FILE = PROCESSED_DIR / "uxbridge_lounge_sales_clean.csv"  # output of data_cleaning.py
CLEANING_LOG_FILE = DOCS_DIR / "cleaning_log.csv"  # what the cleaning changed
FORECAST_MODEL_FILE = MODELS_DIR / "forecast_model.joblib"  # main demand model
STOCKOUT_MODEL_FILE = MODELS_DIR / "stockout_model.joblib"  # stockout yes/no classifier
METADATA_FILE = MODELS_DIR / "metadata.json"  # versions, seed, metrics, menu
METRICS_FILE = DOCS_DIR / "metrics.json"  # all evaluation results
MODEL_CARD_FILE = DOCS_DIR / "model_card.md"  # governance document

# ---------- Reproducibility ----------
SEED = 42  # same seed everywhere = same results every run

# ---------- Time split (train on the past, test on the future) ----------
TRAIN_END = "2026-02-28"  # last day the model is allowed to learn from
TEST_START = "2026-03-01"  # first day used to test it (Mar to Jun 2026 = 4 months)

# ---------- Business settings ----------
SERVICE_LEVEL = 0.6  # default buffer for FRESH items: prep covers demand on about 60% of days.
#   A business choice, not a model setting: see the trade-off chart in docs/figures/.
BOTTLE_SERVICE_LEVEL = 0.9  # bottles keep, so stock generously (no waste cost)
INTERVAL_LOW, INTERVAL_HIGH = 0.10, 0.90  # 80% prediction range: 10th to 90th percentile

# ---------- The official menu (source of truth) ----------
MENU = [
    "Butter Chicken", "Dal Makhani", "Chole Bhature", "Paneer Tikka",  # Punjabi
    "Dhokla", "Undhiyu", "Thepla with Chutney", "Khandvi",  # Gujarati
    "Vada Pav", "Misal Pav", "Pav Bhaji", "Puran Poli",  # Marathi
    "Masala Dosa", "Idli Sambar", "Medu Vada", "Chicken Chettinad",  # South Indian
    "Mango Lassi", "Masala Chai", "Chaas", "Sol Kadhi", "Filter Coffee",  # soft drinks
    "Kingfisher Lager", "Cobra Beer", "House Red Wine 175ml",  # bar
]
WEATHER_TYPES = ["Sunny", "Cloudy", "Rainy", "Snow"]  # allowed weather values
FESTIVALS = ["None", "Pongal", "Holi", "Vaisakhi", "Ganesh Chaturthi",
             "Navratri", "Diwali", "Christmas and New Year"]  # allowed festival values
LOCAL_EVENTS = ["None", "India cricket match"]  # allowed event values
