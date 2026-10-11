# Ames House Price Prediction (Ridge + Feature Engineering)

Arrowstack Technologies virtual internship, Machine Learning track (Batch 2026), mandatory task 1.
Author: Sherifdeen Bilal Olamilekan

> All results are observations on a public dataset. They are decision-support evidence, not production or real-world valuation claims.

## Problem and stakeholder
The goal of this project is to estimate the sale price of a recorded residential property transaction using property attributes and sale-record fields contained in the Ames dataset. The model is intended as a decision-support tool for housing analysts, students, and researchers who want to study predictive modelling on structured housing data.

This is not a pre-listing valuation model because it uses sale-record variables such as SaleType, SaleCondition, MoSold, and YrSold. It should not be used for lending, appraisal, insurance, taxation, or investment decisions.

## Data
- Source: OpenML, `data_id=42165` ("house_prices"), version 1, uploaded 2019-10-04
- Accessed: 2026-10-10
- License: the OpenML license field reads "NA" (not stated); the original source has not been verified here.
- Size: 1,460 rows x 81 columns; target `SalePrice` (dollars)
- Split: random 80/20, seed 42 -> 1,168 train / 292 test. The test set was evaluated once.
- No raw data is stored in this repo; the notebook downloads it by ID.

## Method (final model)
- `Id` dropped. Target modeled as log1p(SalePrice); predictions converted back with expm1, so errors are in dollars.
- Preprocessing inside the pipeline (fit on training folds only): numeric columns median-imputed and standardized; categorical columns filled with a constant "missing" and one-hot encoded.
- Engineered feature: `LogArea = log1p(GrLivArea)`.
- Model: Ridge, alpha = 10 (chosen from {1, 3, 10, 30, 100} on the same CV folds).
- Outlier policy: three training houses with `GrLivArea` > 4,000 were excluded from training only; they are still scored in CV.
- Validation: 5-fold shuffled CV on the training split (seed 42).

## How to reproduce
1. Open the notebook in Google Colab (or any Python 3.13 environment with the packages in `requirements.txt`; internet access is needed to download from OpenML).
2. Runtime > Run all.
3. Check values you should see: median baseline CV MAE 54,586; Ridge baseline 17,313; Ridge + LogArea 16,224; final test MAE 14,313 and median absolute error 10,100.
4. The guard cell at the end of the notebook is a post-test extension and does not change any reported number.

## Results

### Cross-validation, all rows in training (model selection)
| Model / feature set | CV MAE |
|---|---:|
| Median baseline | 54,586 |
| Ridge baseline | 17,313 |
| Ridge + LogArea | 16,224 |
| Ridge + qual_ord | 17,291 |
| Ridge + LogArea + qual_ord | 16,214 |
| Ridge + LogArea + interaction | 16,218 |
| Random Forest | 18,363 |
| Gradient Boosting | 15,916 |

Fold spread (std): Ridge baseline about 1,541; Random Forest about 841; Gradient Boosting about 1,242. Random Forest and Gradient Boosting used raw features and a log target.

### Cross-validation, three houses > 4,000 sq ft excluded from training (all rows still scored)
| Model | CV MAE |
|---|---:|
| Ridge + LogArea | 15,729 |
| Gradient Boosting (raw features) | 15,894 |

Per-fold spread was not measured for this protocol. For scale, the all-rows runs had a fold spread of about 1,541 (Ridge) and 1,242 (Gradient Boosting), much larger than the 165-point difference here, so the two models are treated as comparable.

### Test results (run once)
| Model | Test MAE | Test median AE |
|---|---:|---:|
| Median baseline | 59,568 | 40,000 |
| Final Ridge | 14,313 | 10,100 |

- Test set: 292 houses; only 1 is above 4,000 sq ft.
- Test MAE (14,313) is lower than CV MAE (15,729); with 292 houses this is plausible sampling variation, not evidence of improvement.
- The out-of-range failure was observed in cross-validation and was not meaningfully validated on the test set.
- Gradient Boosting was not evaluated on the test set; the Ridge vs Gradient Boosting comparison rests on CV only.

### Footnote on reported MAE values
Two very similar MAE values appear in this report for the same feature set:
- 16,224 is the mean of the five fold-level MAEs.
- 16,226 is the pooled out-of-fold MAE across all validation predictions.

The small difference occurs because the folds are not exactly equal in size. Both statistics describe the same model and lead to the same conclusions.

## Error analysis
Out-of-fold errors, all rows in training (Ridge + LogArea vs Gradient Boosting, bands by true price quartile):

| Price band | Ridge MAE | GB MAE | Ridge % error |
|---|---:|---:|---:|
| Q1 low | 11,360 | 12,266 | 11.6 |
| Q2 | 14,220 | 10,854 | 9.5 |
| Q3 | 14,256 | 12,909 | 7.7 |
| Q4 high | 25,077 | 27,665 | 8.1 |

Three houses above 4,000 sq ft were Ridge's three worst errors:

| Row | GrLivArea | OverallQual | SaleCondition | SalePrice |
|---|---:|---:|---|---:|
| 1182 | 4,476 | 10 | Abnorml | 745,000 |
| 523 | 4,676 | 10 | Partial | 184,750 |
| 1298 | 5,642 | 10 | Partial | 160,000 |

Sensitivity to those three houses (CV MAE / median AE):

| Protocol | Ridge | Gradient Boosting |
|---|---|---|
| All rows in training | 16,226 / 10,509 | 15,918 / 10,017 |
| Three houses excluded from training | 15,729 / 9,342 | 15,894 / 9,967 |

When trained without the three houses, average error on those same houses was about 907,739 (Ridge) vs 374,784 (Gradient Boosting).

### Interpretation
The largest Ridge errors occurred on three houses with living area greater than 4,000 sq ft. Two of these sales were consistent with the Partial sale condition and unusually low sale prices relative to their size and quality, while the third was a legitimately expensive house sold under an Abnorml sale condition.

When those houses were included in training, Ridge's three worst errors were approximately $914k, $366k, and $266k, averaging about $515k. A training-only exclusion rule for houses above 4,000 sq ft reduced cross-validated MAE from 16,226 to 15,729. However, the threshold was chosen after inspecting model errors, so the result should be treated as directional evidence rather than a confirmed improvement.

The exclusion did not remove the underlying failure mode. When the model was trained without those houses, its average error on the same three houses increased to approximately $908k. This is an observed example of poor performance outside the training range rather than a general conclusion about all large houses.

## Model choice
Selection evidence (CV, training split): Ridge MAE 15,729 vs Gradient Boosting 15,894, comparable within the fold spread (about 1,200 to 1,500). Median absolute errors were 9,342 vs 9,967 (directional). Ridge is simpler and its coefficients can be inspected (with the caveats below), but it produced very large errors on three houses above 4,000 sq ft when they were held out of training (about 907,739 average vs 374,784 for Gradient Boosting).

Final check (test, run once, Ridge only): MAE 14,313 vs 59,568 for the median baseline; median absolute error 10,100. Gradient Boosting was not scored on the test set. Only one test house exceeded 4,000 sq ft, so the out-of-range failure is documented from CV, not test.

Decision: Ridge was chosen because CV performance was comparable to Gradient Boosting, it is simpler to inspect, and its known failure mode is documented and covered by a notebook guard (a post-test extension, below).

## Model inspection
Coefficients from the final Ridge fit (three houses excluded from training), in log-price units; numeric features are standardized (effect per 1 SD), categorical features are one-hot (effect of being in the category).

- Largest positive: LogArea 0.083, Functional_Typ 0.079, Neighborhood_Crawfor 0.078, Neighborhood_StoneBr 0.069, Exterior1st_BrkFace 0.069, OverallQual 0.058, YearBuilt 0.055, OverallCond 0.046.
- Largest negative: MSZoning_'C (all)' -0.057, Neighborhood_MeadowV -0.056, Functional_Maj2 -0.054, SaleCondition_Abnorml -0.052, Heating_Grav -0.051, BldgType_Twnhs -0.039, Condition1_RRAe -0.036, BsmtCond_Fa -0.035.

The largest Ridge coefficients had plausible signs: larger living area, higher overall quality, newer construction, favourable neighbourhoods, and typical functional condition were associated with higher predicted prices, while abnormal sale conditions and certain low-quality categories were associated with lower predicted prices. However, coefficient magnitudes should be interpreted cautiously because numeric and categorical features use different scales, correlated variables share weight, and some categories are rare. The coefficients are useful for inspection and explanation, not for causal conclusions.

## Post-test extensions (model of record unchanged)

### Sale-record fields
A post-test experiment evaluated whether the model depended on sale-record variables (SaleType, SaleCondition, MoSold, and YrSold). Removing these variables changed CV MAE from 16,224 to 16,288 (+64, well inside the approximately +/-1,500 fold spread), indicating no measurable difference. This suggests that a pre-listing variant may be feasible, but it was not validated on the test set and is not the model of record.

### Input guard
A guard in the notebook returns no prediction (NaN) when GrLivArea > 4,000 (1 of 292 test houses). The reported test MAE of 14,313 was computed before the guard was added and includes that house. No deployed enforcement exists outside the notebook.

## Decisions and rejected experiments
Full dated record in `work_log.md`. Rejected: HouseAge (no change), qual_ord (-22, redundant with OverallQual), OverallQual x LogArea (16,218 vs 16,224, mixed folds), Random Forest (worse than Ridge). LogArea helped Ridge but not Gradient Boosting (about +77 change, no measurable difference).

## Limitations
- Three houses above 4,000 sq ft produced extremely large Ridge errors in cross-validation (about 907,739 average when excluded from training).
- The 4,000 sq ft rule was chosen after examining model errors. An independent justification from the original Ames documentation has not been verified.
- Gradient Boosting was not evaluated on the test set.
- Feature and model decisions used the same CV folds, which may add mild selection optimism; alpha was also tuned on those folds.
- A single random split was used (no time-aware validation, although sale year is available).
- The model uses sale-record fields, so it is not a pre-listing estimator.
- The OpenML license field reads "NA".

## Not for use
- Do not use this model for houses with GrLivArea > 4,000 sq ft. A notebook guard returns no prediction (NaN) for these inputs, but there is currently no deployed enforcement mechanism outside the notebook.
- Do not use this model as a production property valuation system or for financial, lending, insurance, or legal decisions.
- Do not interpret predictions as market value guarantees; results were observed on a public dataset and evaluated on a single train/test split.
- Do not use model coefficients to infer causal relationships between housing features and sale price.

## Files
- `README.md` this file
- `model_card.md` short model card
- `work_log.md` dated decision log
- `acceptance_criteria.md` requirements-to-evidence checklist
- `requirements.txt` pinned packages
- the notebook
