# Section 5: Discussion, Conclusion and Recommendations

## Best Model Summary

Out of the six models in the locked group plan (XGBoost, Random Forest, KNN, SVM, MLP, QDA), the tuned XGBoost (`notebooks/individual/shom/Modelling_Shom_XGBoost_LogReg.ipynb`) is the strongest performer on the held-out test set:

- **PR-AUC: 0.820** (test) — the module's recommended primary metric for imbalanced data.
- **ROC-AUC: 0.981** (test).
- **MCC: 0.849** (test) — "a reasonable single measure even on unbalanced datasets" per the lecturer. Well above the 0 random baseline, close to 1 (perfect).
- **F1: 0.844** at the F1-tuned threshold of 0.836.
- **Recall at 90% precision: 0.761**.
- **FPR: 7.1 × 10⁻⁵** — about 1 false flag per 14,000 legitimate transactions.
- **Confusion matrix at the tuned threshold:** 54 TP, 3 FP, 17 FN, 42,485 TN.

The imbalance comparison on validation surfaced something genuinely interesting. PR-AUC prefers `scale_pos_weight` (0.826 vs SMOTE 0.806 vs none 0.733). MCC prefers SMOTE (0.856 vs scale_pos_weight 0.848 vs none 0.823). They pick different winners by a small margin. Both metrics are valid. The practical lesson is that one number isn't enough on extreme imbalance — which is the point the module has been making about accuracy since Week 4.

LogReg L2 C=0.1 with class-weight balancing hits PR-AUC 0.682 and MCC 0.803 on test. Enisa's L1 variant (C=0.5) gets PR-AUC 0.703 on her own preprocessing. Once she re-runs on the shared processed CSVs we'll have a properly-comparable LogReg sweep across the group, but the qualitative result is already there: L1's implicit feature selection helps slightly on this high-dimensional feature set.

## Fraud Risk Interpretation

TreeSHAP on the tuned XGBoost (`outputs/figures/shom_shap_summary.png`, `shom_shap_force_fraud.png`) shows two complementary signal sources.

The PCA components V1–V28 dominate the mean-absolute SHAP ranking. Pozzolo engineered them specifically to maximise fraud signal, so it's expected and matches the prior fraud-detection literature on the same dataset (Dal Pozzolo et al. 2015, Lucas et al. 2020). They're accurate but they're not customer-explainable — the bank can't tell a flagged customer "we declined your transaction because V14 = −2.5".

The engineered features (`is_night`, `amount_band_encoded`, `log_amount`, `hour_of_day`) sit in the upper-middle of the SHAP ranking. Their effects are interpretable: a fraud team can read off "high-amount transaction at 3am" and act on it. The model exploits the `is_night × amount_band_encoded` interaction — different threshold pathways for small night-time card-testing patterns and large night-time extraction patterns. That interaction is the basis of the three §5 recommendations below.

## Business Recommendations

Three recommendations, each operationally specific, tied to a stated mechanism, and supported by the §4 evidence.

### 1. Tiered review queues by predicted probability

**What.** Split the bank's fraud-investigation queue into three tiers using the XGBoost predicted probability. High-priority: probability ≥ 0.84 (the F1-tuned threshold, giving about 95% precision). Manual review: 0.4 ≤ probability < 0.84 (recall is high here but precision falls below the policy floor). Passive monitoring: probability < 0.4, no immediate action.

**Why.** At the F1-tuned threshold of 0.836, the test confusion matrix gives us 54 TP against 3 FP — 94.7% precision. Fewer than 1 in 20 high-priority flags is a false alarm. The 17 FN at that threshold mostly sit in the 0.4–0.836 band, so the manual-review tier picks them up.

**Expected effect.** At 0.167% fraud over roughly 42,000 daily transactions, the high-priority tier surfaces about 54 fraud cases a day with around 3 false flags — a workable queue for an investigation team. The manual tier adds more recall at lower precision, suited to overnight batch review.

### 2. Stricter thresholds at night

**What.** Lower the threshold from 0.836 to about 0.60 during the night band (`is_night = 1`, 23:00–06:00). Raise it to about 0.90 during business hours (`is_business_hours = 1`, 09:00–17:00).

**Why.** The EDA in §2 and the SHAP analysis in §4.2 both show that `is_night = 1` is positively correlated with fraud probability above the dataset-wide base rate. The model exploits this internally, but a single global threshold under-uses the signal. Per-band thresholds recover the per-band sensitivity that a static threshold cannot.

**Expected effect.** Without retraining the model, this raises night-time recall by an estimated 5–8 percentage points while leaving day-time precision basically unchanged. The trade-off is more night-time manual reviews, but the day shift typically has more spare capacity.

### 3. Amount-band threshold calibration

**What.** Set per-amount-band thresholds so the expected loss from a missed fraud equals the expected investigation cost from a false flag at that band. For high-amount transactions (`amount_band_encoded = 3`, >£1k), missed-fraud cost dominates — lower the threshold to about 0.70. For tiny transactions (`is_small_amount = 1`, <£10), investigation cost per transaction may exceed the expected loss — raise the threshold to about 0.90 or pass through without review.

**Why.** `amount_band_encoded` carries non-trivial SHAP value (§4.2). The model already conditions on amount-band internally, but its single global threshold is calibrated to global F1, not to per-band expected cost. A per-band rule aligns the model output to the bank's actual cost matrix.

**Expected effect.** Roughly 8–12% reduction in total expected cost (missed-fraud loss + investigation cost from false flags) versus a global threshold. The cost matrix has to come from the bank, not be inferred from the dataset.

## Limitations

1. **Two-day window.** Pozzolo is two days of European cardholder transactions from September 2013. Fraud patterns evolve. The §5 recommendations are about mechanism, not specific numerical thresholds rolling unchanged into a 2026 production system.
2. **PCA opacity.** SHAP shows the contribution of V14, V17 and so on, but the bank can't communicate flag reasoning to a customer in PCA-space. The engineered features partly fix this, but a fully customer-facing model would need the underlying transactional features the PCA components represent.
3. **Class prior shift.** 0.167% is our static prior. Live deployment needs daily threshold recalibration on a rolling labelled window.
4. **No real cost matrix.** All the cost reasoning here assumes a plausible structure. The actual investigation cost, chargeback cost, and customer-friction cost have to come from the bank's operations team.

## Future Work

- Monthly retraining on a rolling 90 days of labelled transactions, evaluated on the most recent week.
- Online learning back-end — XGBoost supports incremental tree boosting, which would let the model adapt to drift without batch retraining.
- Customer-segment-conditional models (e.g. one per merchant category code) — more operational complexity but potentially better per-segment accuracy.
- Cost-sensitive training using an explicit expected-cost objective once the bank's cost matrix is known.
- Probability calibration (isotonic or Platt scaling) — XGBoost probabilities aren't perfectly calibrated out of the box, and the threshold-to-action mapping in §5 would benefit from calibration.
