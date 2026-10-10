# Work log: Ames house price prediction

2026-10-10: Added LogArea, CV MAE improved from 17,313 to 16,224 (+6%), all five folds positive.
2026-10-10: Tested HouseAge, CV MAE unchanged, rejected.
2026-10-10: Tested qual_ord, CV MAE 17,291 (-22 vs baseline), redundant with OverallQual, rejected.
2026-10-10: Tried OverallQual x LogArea interaction, CV MAE 16,218 vs 16,224, mixed folds, rejected.
2026-10-10: Ridge alpha sweep, alpha=10 best. Random Forest scored 18,363 vs Ridge's 17,313 baseline, worse.
2026-10-10: Found three flagged houses as Ridge's worst errors; two consistent with Partial sale condition, one genuine high-price anomaly.
2026-10-10: Tested outlier policy: excluding >4,000 sq ft homes from training improved Ridge's overall MAE more than GB's, though Ridge remained 2.4x worse on flagged houses. Caveat: threshold chosen after seeing worst errors; will verify Ames documentation for independent justification.
2026-10-10: Final test evaluation: Ridge MAE 14,313 vs baseline 59,568, median error 10,100, with 1 test house >4,000 sq ft.

TODO (add in your own words, with the date): data pinned to OpenML 42165; baselines (median 54,586, Ridge 17,313); Gradient Boosting 15,916 and no measurable change with LogArea; error analysis by price band; decision to choose Ridge and why; clean notebook restarted and run top to bottom with matching numbers.
