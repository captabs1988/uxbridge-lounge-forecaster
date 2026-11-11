# Phase 7: Governance, Ethics and Risk

## 7.1 Explainability
- **Global:** permutation importance shows what drives forecasts overall: the menu item, cuisine,
  weekend, day of week, then weather, cricket and temperature (`figures/feature_importance.png`).
- **Local:** SHAP shows how each clue pushes a single forecast up or down (`figures/shap_summary.png`).
- **For the chef:** the app shows a likely range and plain-English drivers, not just a number.

## 7.2 Fairness and bias
There are no people in this data, so "fairness" here means **the model must work equally well for every
part of the menu**. If it were worse for, say, Gujarati dishes, those customers would face more stockouts
and those dishes more waste.
- Measured: % error (WAPE) for food vs drink, each cuisine, each price tier.
- Rule: flag any group with error more than 1.25x the overall error.
- Result: no group flagged (all within 0.92x to 1.08x). See `metrics.json > fairness`.
- Known gap: festival days have about 2x the normal error (only 48 test rows, one year of history).

## 7.3 Data privacy (UK GDPR)
- The dataset contains **no personal data**: no customer names, card details, staff IDs or contact details.
- It is synthetic, so no consent or data sharing agreement is needed.
- **If real data is used later:** use aggregated till data only (item-level daily totals), never
  customer-level transactions; agree a lawful basis with the data controller; keep data in the UK/EEA;
  set a retention period (e.g. 3 years); limit access to named roles; record it in the
  organisation's data processing register.

## 7.4 Legal and regulatory context
- **UK AI regulation principles** (safety, transparency, fairness, accountability, contestability):
  the model card, explainability charts and "chef has the final say" design cover these.
- **Natasha's Law (UK allergen labelling):** this model does NOT handle allergens. Prep quantities must
  never change recipes or labels. Allergen procedures stay exactly as they are.
- **Food safety:** the model may recommend more prep on busy days; existing hot-holding and
  chilling times still apply.

## 7.5 Risk assessment
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Forecast too low on a surprise busy day (stockouts) | Medium | Medium | Likely range shown; stockout alerts; chef can override |
| Forecast too high (waste) | Medium | Low | Buffer setting is adjustable; waste tracked in monitoring |
| Wrong weather forecast input | Medium | Low | Tested: 2C error raises MAE only from 1.04 to 1.09 |
| Model drifts as menu or customers change | Medium | Medium | Monitoring panel; retrain monthly; alert if error grows 25% |
| Over-reliance: staff stop using judgement | Medium | Medium | Decision support only; chef signs off the prep list |
| New festival or menu item not in training data | Low | Medium | Input checks reject unknown values; add data then retrain |
| Synthetic data doesn't match a real restaurant | High | High | Pilot on real POS data before live use (see roadmap) |

## 7.6 Impact on staff
The tool **supports** the chef, it doesn't replace him. Prep decisions, recipes and staffing stay with people.
It removes guesswork, not jobs. Kitchen staff get a clearer prep list, and the chef's knowledge
(e.g. a local event the model doesn't know) can always override it.

## 7.7 Model card
See `docs/model_card.md` (generated automatically on every training run).

## 7.8 Reproducibility
- Seed 42 in data generation, model training, tuning and tests.
- `requirements.txt` lists the libraries; `models/metadata.json` records exact versions used.
- The whole project rebuilds with 3 commands (see README), and 25 automated tests check it.
