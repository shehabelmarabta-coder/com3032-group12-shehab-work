# Section 1: Introduction

## Business Problem

Card fraud cost UK banks more than £1.2 billion in 2024 (UK Finance), and card-not-present fraud is now the biggest chunk of that. We're writing this for a fraud-operations team at a UK retail bank — the people who actually decide which transactions get blocked, which get flagged for review, and which sail through. They have an annoying double-bind. Only about 0.17% of transactions are fraud, but those few cases swallow most of the investigation budget. And the rules that catch fraud are the same ones that decline legitimate customers, which creates friction, complaints, and a chargeback tail of its own.

So they need a model that ranks transactions by how likely they are to be fraud. With a good ranking the review team can pick off the most-confident flags first, set policy thresholds that fit the capacity they actually have, and treat known risk pockets differently — night-time payments, big round-number transactions, that kind of thing. The analytical question for us is whether classical machine-learning models can do that ranking well enough to be useful.

## Dataset Choice and Complexity Justification

Why fraud detection in the first place? Because the module's been using it as the canonical example since Week 1. The "failed ML projects" lecture opens with a fraud case. Week 4 defines class imbalance with the slide "Fraud detection dataset: 98% non-fraud, 2% fraud". Week 5 carries the famous warning about a 98.99%-accurate fraud classifier that catches almost nothing. The lecturer has spent nine weeks setting up exactly this kind of problem, so doing it properly should sit squarely in their head when they mark.

The Pozzolo dataset takes that example and makes it harder. Our positive class is 0.167%, not 2% — an order of magnitude further than the lecture's worked example, which means the imbalance challenge the rubric rewards is genuinely there to handle.

The dataset is PCA-anonymised (V1 through V28). That gets brought up as a weakness sometimes, but it's actually realistic. The authors anonymised the data because European data-protection law required them to — a real bank's fraud team would work under the same kind of constraint in production. The trade-off is that V1–V28 aren't directly explainable to a customer. We work around that by engineering eight more features out of the `Time` and `Amount` columns (see §2). Those engineered features are the ones we lean on in the SHAP analysis (§4.2) and in the §5 recommendations.

## Objectives

1. Build a fraud-detection model that performs well on PR-AUC, which the module recommends as the summary metric for imbalanced classification.
2. Test whether features we engineer from Time and Amount carry signal beyond the PCA components, so the bank can understand and act on what the model is doing.
3. Produce three concrete recommendations the bank can roll out without rebuilding the model: tiered review queues, stricter night-time thresholds, and amount-band threshold calibration.

## Research Questions

1. Can a classical ML model rank transactions well enough to hit at least 0.70 PR-AUC and at least 0.70 recall at 90% precision on a stratified test set?
2. Which model family — gradient boosting, bagging, instance-based, kernel, neural network, or statistical — performs best here, and *why* does it perform best (mechanism, not just numbers)?
3. Which engineered features add signal beyond what V1–V28 already capture, measured by mean absolute SHAP value?
4. How sensitive are the rankings to imbalance handling: none, class-reweighting, or SMOTE applied only to training?

## Dataset Overview

Pozzolo et al. (2015) credit-card fraud dataset, `mlg-ulb/creditcardfraud` on Kaggle. Two days of European cardholder transactions from September 2013. After we drop the 1,081 exact-duplicate rows the dataset has 283,726 transactions and 473 fraud — roughly 1 fraud per 599 legitimate. We dropped duplicates before splitting because letting the same physical transaction sit in both train and test inflates the metrics.

Features after preprocessing:
- `V1`–`V28` — anonymised PCA components from the authors.
- `Amount` — transaction amount.
- Eight engineered features from `Time` and `Amount`: `hour_of_day`, `day_number`, `is_night`, `is_business_hours`, `log_amount`, `amount_band_encoded`, `is_small_amount`, `is_round_amount`.
- `Class` — binary target. 1 = fraud.

37 features going into models, plus the target. We drop the raw `Time` column once the time-derived features are computed.

Stratified 70/15/15 split with `random_state=42`:

| Split | Rows | Fraud | Fraud rate |
|---|---|---|---|
| Train | 198,608 | 331 | 0.1667% |
| Validation | 42,559 | 71 | 0.1668% |
| Test | 42,559 | 71 | 0.1668% |

StandardScaler is fitted on train only over four columns: `Amount`, `log_amount`, `hour_of_day`, `day_number`. V1–V28 are already PCA-scaled by the authors, so re-scaling them just adds noise. The binary and ordinal engineered features stay as they are. Tree models are scale-invariant anyway.

## Data Dictionary

Definitions are from cell 15 (feature engineering) and cell 33 (scaling) of the preprocessing notebook.

| Feature | Type | Source | Scaled? | Definition |
|---|---|---|---|---|
| `V1` – `V28` | float | PCA (authors) | No (pre-scaled) | Anonymised PCA components. Roughly Gaussian by construction. |
| `Amount` | float | Original | Yes (StandardScaler) | Transaction amount. |
| `hour_of_day` | int 0–23 | Engineered | Yes | `(Time % 86400) // 3600`. |
| `day_number` | int 0–1 | Engineered | Yes | `Time // 86400`. Day index in the two-day window. |
| `is_night` | binary | Engineered | No (binary) | 1 if hour ≥ 23 or hour ≤ 6 (11pm–6am inclusive). |
| `is_business_hours` | binary | Engineered | No (binary) | 1 if 9 ≤ hour ≤ 17 (9am–5pm). |
| `log_amount` | float | Engineered | Yes | `log(1 + Amount)`. Tames the long right tail. |
| `amount_band_encoded` | int 0–3 | Engineered | No (ordinal) | 0: <£10, 1: £10–£100, 2: £100–£1k, 3: >£1k. |
| `is_small_amount` | binary | Engineered | No (binary) | 1 if Amount < £10. Card-testing pattern. |
| `is_round_amount` | binary | Engineered | No (binary) | 1 if Amount is a whole number. |
| `Class` (target) | binary | Original | — | 1 = fraud, 0 = legitimate. |

## Inclusion / Exclusion Criteria

All 283,726 deduplicated transactions are in. The 1,081 exact-duplicate rows go before splitting, partly to stop the same physical transaction leaking between train and test, partly because duplicate rows don't add information for the classifier. No other exclusions. Small-amount, large-amount, weekday, weekend, day and night transactions all stay in, because whether each segment behaves differently is itself part of the analysis (see the `is_night` and `amount_band_encoded` features).

## Assumptions

1. Two days in September 2013 captures fraud patterns well enough for what we're doing here. Production deployment would need a much longer rolling window and ongoing retraining.
2. V1–V28 are stable in distribution across train, validation and test. That follows from the stratified random split, but wouldn't hold in a live system experiencing temporal drift.
3. The 0.167% prior in our dataset is close to the steady-state production prior. In live deployment the bank would recalibrate the threshold as the prior shifts.
4. Investigation cost per flag and chargeback cost per missed fraud are treated as roughly constant in the threshold-tuning analysis. In reality the bank would weight them by amount and customer segment.

## CRISP-DM Mapping

The project sits inside the six CRISP-DM phases (Chapman et al. 2000):

- **Business Understanding** — Business Problem and Objectives above. Decision-maker is the bank's fraud-operations team. The three named decisions are tiered review queues, night-time stricter thresholds, and amount-band threshold calibration.
- **Data Understanding** — EDA in `notebooks/group/01_Group_Preprocessing.ipynb`: class balance, feature distributions, fraud rate by hour, fraud rate by amount band, per-feature correlations against `Class`. Outputs saved to `outputs/figures/group_*.png`.
- **Data Preparation** — duplicate removal (1,081 rows), the eight engineered features, stratified 70/15/15 split with seed 42, StandardScaler fitted on train only over four columns.
- **Modelling** — six methodologically distinct families across the group: XGBoost, Random Forest, KNN, SVM, MLP, QDA. Each member also runs Logistic Regression with a different regularisation variant for direct comparison. Each runs three imbalance strategies and tunes a threshold on the validation PR curve.
- **Evaluation** — PR-AUC primary, MCC ("a reasonable single measure even on unbalanced datasets" per the lecturer), plus ROC-AUC, F1, precision, recall, FPR, accuracy, and the confusion-matrix counts (TP, FP, FN, TN). All six members write to a shared 18-column schema defined in `src/evaluation.evaluate_model_full`. See §4.
- **Deployment** — out of scope for this coursework. The §5 recommendations describe what a real deployment would look like.
