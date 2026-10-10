# Model card: Ames house price model (Ridge + LogArea)

DRAFT: resolve every TODO before publishing.

- Purpose and intended user: TODO (decision-support price estimate; who uses it)
- Data: OpenML 42165 v1, accessed 2026-10-10, license "NA" on OpenML. 1,460 rows; 1,168 train / 292 test.
- Method: Ridge (alpha 10) on log1p(SalePrice), LogArea feature, three houses over 4,000 sq ft excluded from training.
- Evidence: see README (CV tables and one-time test result are reported separately).
- Failure modes: large errors outside the training range (living area above 4,000 sq ft).
- Not for: see README "Not for use".
- Next steps: TODO (your words; for example time-aware validation, input guard, tuning Gradient Boosting)
