# Model card: Ames house price model (Ridge + LogArea)

- Purpose and intended user: estimate the sale price of a recorded residential transaction; decision-support for housing analysts, students and researchers studying predictive modelling. Not a pre-listing valuation model (it uses SaleType, SaleCondition, MoSold and YrSold).
- Data: OpenML 42165 v1, accessed 2026-10-10, license "NA" on OpenML. 1,460 rows; 1,168 train / 292 test.
- Method: Ridge (alpha 10) on log1p(SalePrice), LogArea feature, three houses over 4,000 sq ft excluded from training only.
- Evidence: see README (CV tables and the one-time test result are reported separately). Test MAE 14,313 vs 59,568 for a constant-median baseline.
- Failure modes: very large errors outside the training range (living area above 4,000 sq ft), observed in cross-validation; a notebook guard returns no prediction for these inputs (post-test extension, no deployed enforcement).
- Not for: see README "Not for use".
- Next steps: validate the 4,000 sq ft rule independently, move the input guard from a notebook function into a prediction service, and evaluate additional models under the same validation design.
