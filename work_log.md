# Work log: Ames house price prediction

- 2026-10-10: Pinned the dataset to OpenML data_id 42165 (version 1); random 80/20 split, seed 42 (1,168 train / 292 test).
- 2026-10-10: Baselines (5-fold CV, log target): median baseline MAE 54,586; Ridge (alpha 10) 17,313.
- 2026-10-10: Added LogArea, CV MAE improved from 17,313 to 16,224 (+6%), all five folds positive.
- 2026-10-10: Tested HouseAge, CV MAE unchanged, rejected.
- 2026-10-10: Tested qual_ord, CV MAE 17,291 (-22 vs baseline), redundant with OverallQual, rejected.
- 2026-10-10: Tried OverallQual x LogArea interaction, CV MAE 16,218 vs 16,224, mixed folds, rejected.
- 2026-10-10: Ridge alpha sweep, alpha=10 best. Random Forest scored 18,363 vs Ridge's 17,313 baseline, worse. Gradient Boosting (raw features) scored 15,916; adding LogArea changed it by about 77, no measurable difference.
- 2026-10-10: Error analysis by price quartile: Ridge MAE 11,360 / 14,220 / 14,256 / 25,077; Gradient Boosting 12,266 / 10,854 / 12,909 / 27,665.
- 2026-10-10: Found three flagged houses as Ridge's worst errors; two consistent with Partial sale condition, one genuine high-price anomaly.
- 2026-10-10: Tested outlier policy: excluding >4,000 sq ft homes from training improved Ridge's overall MAE more than GB's, though Ridge remained 2.4x worse on flagged houses. Caveat: threshold chosen after seeing worst errors; will verify Ames documentation for independent justification.
- 2026-10-10: Decision: Ridge + LogArea chosen over Gradient Boosting (CV comparable, 15,729 vs 15,894; Ridge simpler to inspect).
- 2026-10-10: Final test evaluation: Ridge MAE 14,313 vs baseline 59,568, median error 10,100, with 1 test house >4,000 sq ft.
- 2026-10-10: Clean notebook restarted and run top to bottom; printed numbers matched this log.
- 2026-10-10: Post-test extension: removed sale-record fields (SaleType, SaleCondition, MoSold, YrSold); CV MAE 16,288 vs 16,224 (+64, inside the ~1,500 fold spread), no measurable difference; model of record unchanged.
- 2026-10-10: Inspected Ridge coefficients for interpretability; the largest coefficients had plausible signs, but magnitudes mix scales and are affected by correlation and rare categories; no model changes made.
- 2026-10-10: Added post-test guard returning no prediction (NaN) for GrLivArea > 4000 (1 of 292 test houses); reported test MAE 14,313 was computed before the guard and includes that house.
