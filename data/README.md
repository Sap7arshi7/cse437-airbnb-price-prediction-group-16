# Dataset Documentation

This directory contains the data files used by the **Airbnb Price Prediction Using Machine Learning** project for CSE437, Section 6.

Project repository: [cse437-airbnb-price-prediction-group-16](https://github.com/Sap7arshi7/cse437-airbnb-price-prediction-group-16)

## Dataset Overview

| Item | Details |
|---|---|
| Dataset | London Airbnb Detailed Listings Dataset |
| Provider | Inside Airbnb |
| Source page | [Inside Airbnb: Get the Data](https://insideairbnb.com/get-the-data/) |
| Required source file | London `listings.csv.gz` (Detailed Listings data) |
| Local filename | `data/raw/listings.csv` |
| Raw dataset used in the project | 92,638 rows and 90 columns |
| Prediction target | `log_price` |

The dataset includes property, host, location, room-type, availability, review, amenity, and price information for Airbnb listings in London.

## Download Instructions

1. Open the [Inside Airbnb data-download page](https://insideairbnb.com/get-the-data/).
2. Find **London** in the city list.
3. Download `listings.csv.gz`, which is labelled **Detailed Listings data**. Do not use the smaller `listings.csv` summary file.
4. Extract `listings.csv.gz` with 7-Zip, WinRAR, `gzip`, or another archive tool.
5. Make sure the extracted file is named `listings.csv`.
6. Create the `data/raw/` directory if it does not already exist.
7. Place the extracted file at `data/raw/listings.csv`.

The expected local structure is:

```text
cse437-airbnb-price-prediction-group-16/
└── data/
    ├── raw/
    │   └── listings.csv
    ├── processed/
    ├── README.md
    ├── model_comparison_results.csv
    └── predictions.csv
```

From the project root, confirm that the file is in the correct location before running the notebooks:

```bash
ls data/raw/listings.csv
```

On Windows PowerShell, use:

```powershell
Get-Item data\raw\listings.csv
```

## Why the Raw Dataset Is Not Uploaded

The uncompressed raw dataset is not stored in this repository because it exceeds GitHub's 100 MiB per-file limit. GitHub blocks normal Git uploads above that limit, as described in its [large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Keeping the raw file outside the repository also avoids duplicating a dataset that is already maintained by Inside Airbnb. Each user should download the file from the original provider and keep it locally. The raw file must remain unchanged so that the cleaning and preprocessing steps can be reproduced.

## Important Reproducibility Note

Inside Airbnb updates its downloadable datasets periodically. A newer London file may contain a different number of rows or columns from the snapshot used in this project. The recorded results were produced from a raw file containing 92,638 rows and 90 columns. To reproduce the reported metrics exactly, use the same dataset snapshot when available.

## Processed Data

The notebooks generate the following files:

| File | Description |
|---|---|
| `processed/X_train.csv` | Training predictors: 49,792 rows and 160 features |
| `processed/X_test.csv` | Test predictors: 12,448 rows and 160 features |
| `processed/y_train.csv` | Training values for `log_price` |
| `processed/y_test.csv` | Test values for `log_price` |
| `processed/feature_importance.csv` | Feature-importance values produced by the Gradient Boosting model |
| `model_comparison_results.csv` | Baseline-model evaluation results |
| `predictions.csv` | Final-model predictions and residual errors |

The raw data is processed through these main steps:

- convert the currency-formatted `price` field to a numerical value;
- remove listings with a missing or non-positive price;
- cap target prices at the 99th percentile;
- create `log_price` using a `log1p` transformation;
- remove empty, unsuitable, and leakage-prone fields;
- fill missing numerical values with feature medians;
- fill missing categorical values with `Unknown`;
- encode binary and categorical predictors;
- group less frequent neighbourhoods under `Other`; and
- split the data into 80% training and 20% testing subsets with `random_state=42`.

## Reproducing the Data Pipeline

Install the required libraries from the project root:

```bash
pip install -r requirements.txt
```

Then run the notebooks in this order:

1. `notebooks/01_data_audit_and_eda.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_feature_engineering.ipynb`
4. `notebooks/04_modeling_and_tuning.ipynb`
5. `notebooks/05_evaluation_and_error_analysis.ipynb`

All notebooks use relative paths and should be executed from the `notebooks/` directory or with a working directory configuration that preserves those paths.

## Data Responsibility

The dataset remains the responsibility of its original provider. Users should follow the usage conditions published by Inside Airbnb and cite the source when presenting results based on the data.
