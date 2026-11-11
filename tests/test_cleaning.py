"""Tests for Phase 2: the cleaning must always produce data that passes every business rule."""
import pandas as pd
import pytest

from src import config
from src.data_cleaning import clean_data, validate


@pytest.fixture(scope="module")  # clean once, share with every test in this file (faster)
def cleaned():
    raw = pd.read_csv(config.RAW_SALES_FILE)  # the messy file
    clean, log = clean_data(raw)
    return raw, clean, log


def test_all_validation_rules_pass(cleaned):
    _, clean, _ = cleaned
    failed = [rule for rule, ok in validate(clean).items() if not ok]  # list any broken rules
    assert failed == [], f"Rules failed: {failed}"


def test_raw_data_is_not_changed(cleaned):
    raw, _, _ = cleaned
    assert len(raw) == 13342  # the raw table still has every row: cleaning worked on a copy


def test_enough_rows_kept(cleaned):
    raw, clean, _ = cleaned
    assert len(clean) / len(raw) > 0.95  # we should keep at least 95% of the data


def test_log_has_every_step(cleaned):
    _, _, log = cleaned
    assert len(log) == 17 and log["rows_affected"].ge(0).all()  # 17 steps, no negative counts


def test_text_values_standardised(cleaned):
    _, clean, _ = cleaned
    assert set(clean["item_name"]) == set(config.MENU)
    assert set(clean["stockout"]) == {"Yes", "No"}
