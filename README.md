# Credit Card Fraud Detection for Transaction Risk Management

## Purpose

This project supports the COM3032 Business Analytics with Data Visualisation group coursework. The aim is to analyse transaction data and develop fraud detection models that help identify high-risk fraudulent transactions while balancing financial loss, customer friction, and operational cost.

## Project Structure

```text
data/
  raw/                 Original datasets added by the group
  processed/           Cleaned train, validation, and test datasets
notebooks/
  group/               Shared preprocessing, visualisation, and recommendations notebooks
  individual/          Individual modelling notebooks
src/                   Reusable configuration, evaluation, and plotting code
outputs/
  figures/             Saved charts
  results/             Saved model metrics and comparison tables
reports/
  sections/            Draft notes for report sections
```

## Expected Workflow

1. Add the original transaction dataset to `data/raw/`.
2. Run `notebooks/group/01_Group_Preprocessing.ipynb` to inspect, clean, split, and save the processed datasets.
3. Save processed datasets in `data/processed/` as `train_processed.csv`, `val_processed.csv`, and `test_processed.csv`.
4. Individual notebooks read the processed datasets from `data/processed/`.
5. Shehab's individual notebook focuses on Logistic Regression and Random Forest models.
6. Save figures to `outputs/figures/` and model results to `outputs/results/`.
7. Use `notebooks/group/02_Group_Visualisation_Recommendations.ipynb` to compare group results and prepare business recommendations.

## Running the Notebooks

Install the practical project dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter from the project root:

```bash
jupyter notebook
```

Run the group preprocessing notebook before running individual modelling notebooks, because the individual notebooks expect the processed datasets to already exist.
