# Phase 0: Business Framing

## Problem
Uxbridge Lounge is an Indian restaurant (Punjabi, Gujarati, Marathi and South Indian dishes plus a bar).
Every morning the chef decides how many portions of each fresh dish to prepare, based on experience.

- **Too much prep = waste.** Unsold fresh food goes in the bin. About 7% of everything prepared is wasted,
  roughly GBP 22,600 over 18 months.
- **Too little prep = stockouts.** About 22% of item-days sell out. Customers are disappointed and sales are lost.

The chef plans well for things he knows (weekends, festivals) but can't plan for weather or surprise
cricket matches. That is where most stockouts come from (see EDA insight 7).

## Objectives
1. Forecast daily demand for each of the 24 menu items.
2. Recommend a prep quantity that cuts waste without increasing stockouts.
3. Warn the chef when his own plan has a high risk of running out.

## Baseline KPIs (current practice, chef's decisions)
| KPI | Baseline | Source |
|---|---|---|
| Fresh food waste rate | 7.0% of portions prepared | clean data, all 18 months |
| Waste cost | GBP 5,257 in the test period (Mar to Jun 2026) | `docs/metrics.json` |
| Stockout rate (fresh items) | 23% of item-days in the test period | `docs/metrics.json` |
| Forecast error, "same day last week" method | MAE 1.87 portions | `docs/metrics.json` |

## Expected benefits
- Lower waste cost (simulation: 14% to 33% less at the default buffer).
- Fewer stockouts at the same time (simulation: 13% to 29% vs 23% today).
- Faster, more consistent planning: one screen instead of gut feel.

## Users and stakeholders
| Who | Role | How they're affected |
|---|---|---|
| Head chef | Main user | Gets a daily prep list; keeps the final say |
| Bar manager | User | Bottle stock levels, cricket day alerts |
| Restaurant manager | Approver | Sets the buffer (service level), tracks waste and GP |
| Kitchen staff | Affected | Prep lists change; no change to roles or hours |
| Owner / finance | Sponsor | Lower food cost, better margins |
| Customers | Affected | Fewer "sorry, sold out" moments |

## Scope
In: daily demand forecast per item, recommended prep, stockout risk, waste and revenue estimates.
Out: staffing, pricing, ordering from suppliers, allergen management (stays with existing procedures).

## Data
Synthetic data for a fictional restaurant (no real company data, so no consent needed and no personal data).
Waste levels calibrated to the public Kaggle food wastage dataset. See `docs/data_lineage.md`.
