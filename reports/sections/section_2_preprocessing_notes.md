# Section 2: Data Pre-processing and Exploration

## Data Inspection

The raw Pozzolo dataset (`data/raw/creditcard.csv`) is 284,807 rows by 31 columns. 28 anonymised PCA components (`V1`–`V28`), the original `Time` (seconds since the first transaction), `Amount`, and the binary `Class` target. Everything's numeric, so no categorical encoding to deal with at the input layer. Inspection in cells 4–7 of the preprocessing notebook: data types confirmed, summary statistics printed.

## Missing Values

`df.isnull().sum()` returns zero across every column. Nothing to impute. Pozzolo released the dataset clean, which matches the paper.

## Duplicates

`df.duplicated().sum()` returns 1,081 exact-duplicate rows. 19 of them are fraud rows, so we lose 19 of the original 492 fraud cases and 1,062 legitimate rows — leaving 473 fraud out of 283,726 total. We drop duplicates before the split. If you leave them, the same physical transaction can sit in both train and test, and evaluation metrics get inflated.

## Feature Engineering

This is where we did most of the work. The PCA components (`V1`–`V28`) are already centred and scaled by Pozzolo, so traditional preprocessing on those columns is limited. Nothing to impute, nothing real to encode, no outliers to cap in a meaningful way. The genuine preprocessing surface on this dataset is the `Time` and `Amount` columns, and we leaned hard into them.

Eight features derived from `Time` and `Amount`. We drop the raw `Time` column after engineering — once you have hour-of-day and day-number, the seconds-since-first-transaction column doesn't help any more.

| Feature | Definition | Hypothesis |
|---|---|---|
| `hour_of_day` | `(Time % 86400) // 3600`, int 0–23 | Fraud has a diurnal cycle — literature says it peaks at night. |
| `day_number` | `Time // 86400`, 0 or 1 | Day index inside the two-day window. |
| `is_night` | 1 if hour ≥ 23 or hour ≤ 6 (11pm–6am) | Clean binary split for tree models. Foundation for the night-time rule in §5. |
| `is_business_hours` | 1 if 9 ≤ hour ≤ 17 (9am–5pm) | Complements `is_night` for a three-band time signal. |
| `log_amount` | `log(1 + Amount)` | Tames the long right tail for the non-tree models. |
| `amount_band_encoded` | 0: <£10, 1: £10–£100, 2: £100–£1k, 3: >£1k | Foundation for the amount-band rule in §5. |
| `is_small_amount` | 1 if Amount < £10 | Card-testing pattern — fraudsters try a small charge to confirm a card works. |
| `is_round_amount` | 1 if Amount is a whole number | Round amounts are over-represented in automated fraud. |

37 features going into the models, plus the target.

## Encoding

Nothing to encode. The PCA components, Amount and Time are continuous. Class is binary. The engineered features are either binary or ordinal integer by construction.

## Scaling

StandardScaler fitted on the training split only and applied to validation and test. We scale four columns: `Amount`, `log_amount`, `hour_of_day`, `day_number` (cell 33 of the preprocessing notebook). V1–V28 are already on a comparable scale from Pozzolo's PCA transform, so re-scaling them just adds noise. The binary and ordinal engineered features (`is_night`, `is_business_hours`, `is_small_amount`, `is_round_amount`, `amount_band_encoded`) stay raw because their meaning is in the integer encoding.

Fitting only on train stops distributional information about validation and test leaking into the training procedure. Tree models (XGBoost, Random Forest) are scale-invariant by construction, so the scaling does nothing to them — it's there for the LogReg, KNN, SVM and MLP runs.

## Class Imbalance

473 fraud out of 283,726 total transactions — 0.167%, about 1:599. EDA gives us two operational regularities:

- **Time of day**: fraud rate is meaningfully higher in the night band than in business hours (`outputs/figures/group_fraud_by_hour.png`).
- **Amount band**: both the smallest band and the largest band show elevated fraud share relative to the middle bands (`outputs/figures/group_night_and_band_fraud.png`).

We handle imbalance at the modelling stage, not in preprocessing. Every member tries three strategies in their own notebook: none (model sees the raw imbalance), a class-weighting variant specific to their model (`scale_pos_weight` for XGBoost, `class_weight='balanced'` for LogReg and SVM, `weights='distance'` for KNN, native priors for QDA), and SMOTE applied to training only.

## Train / Validation / Test Split

`train_test_split` from sklearn with `stratify=y`, `random_state=42`, and two sequential splits to land at 70/15/15:

| Split | Rows | Fraud | Fraud rate |
|---|---|---|---|
| Train | 198,608 | 331 | 0.1667% |
| Validation | 42,559 | 71 | 0.1668% |
| Test | 42,559 | 71 | 0.1668% |

The stratified split keeps the class prior within 0.0001 percentage points across the three splits, which matters: an unstratified split could leave a validation or test set with a noticeably different fraud rate, and that would contaminate the threshold-tuning analysis. The seed (`42`) is in `src/config.py` as `RANDOM_STATE`, so every member reproduces the same split.
