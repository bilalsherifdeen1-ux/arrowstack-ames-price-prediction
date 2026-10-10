# Ames House Price Prediction (Ridge + Feature Engineering)

Arrowstack Technologies virtual internship, Machine Learning track (Batch 2026), mandatory task 1.
Author: Sherifdeen Bilal Olamilekan

> All results are observations on a public dataset. They are decision-support evidence, not production or real-world valuation claims.
>
> DRAFT NOTE: every "TODO" below must be resolved (and this note deleted) before publishing.

## Problem and stakeholder
TODO (your words): who would use a price estimate, what decision it supports, and what "success" means here (for example, lower MAE in dollars than a constant-median baseline).

## Data
- Source: OpenML, `data_id=42165` ("house_prices"), version 1, uploaded 2019-10-04
- Accessed: 2026-10-10
- License: the OpenML license field reads "NA" (not stated). TODO: verify and cite the original source before claiming anything more.
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

Interpretation: TODO (your words; use "observed" and "consistent with", not causal claims).

## Decisions and rejected experiments
Full dated record in `work_log.md`. Rejected: HouseAge (no change), qual_ord (-22, redundant with OverallQual), OverallQual x LogArea (16,218 vs 16,224, mixed folds), Random Forest (worse than Ridge). LogArea helped Ridge but not Gradient Boosting.
Model choice: TODO (your justification; keep CV selection evidence and the one-time test result separate).

## Limitations
- Three houses above 4,000 sq ft produced extremely large Ridge errors in cross-validation (about 907,739 average when excluded from training).
- The 4,000 sq ft rule was chosen after examining model errors. TODO: verify whether the dataset documentation gives an independent justification, and cite it.
- Gradient Boosting was not evaluated on the test set.
- Feature and model decisions used the same CV folds, which may add mild selection optimism.
- TODO: add (a) single random split (sale-year range from `YrSold` min/max), (b) alpha tuned on the same folds, (c) license "NA".

## Not for use
- Houses with living area above 4,000 sq ft. TODO: state whether this is enforced in code or only documented.
- A production property valuation system.
- Financial, lending, insurance, or appraisal decisions without additional validation.
- Populations outside the Ames dataset distribution.

## Files
- `README.md` this file
- `model_card.md` short model card
- `work_log.md` dated decision log
- `acceptance_criteria.md` requirements-to-evidence checklist
- `requirements.txt` pinned packages
- the notebook (add it: Colab > File > Download > .ipynb, or Save a copy in GitHub)
