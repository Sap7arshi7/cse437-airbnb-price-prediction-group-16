
# Airbnb Price Prediction Using Machine Learning

GitHub Repository:
https://github.com/Sap7arshi7/cse437-airbnb-price-prediction-group-16

## Dataset Usage

The raw dataset is not included in this repository because the file size exceeds GitHub's 100 MB limit.

The dataset can be downloaded from:

Inside Airbnb:
https://insideairbnb.com/get-the-data/

After downloading, place the file inside:

data/raw/listings.csv

The raw dataset remains unchanged to maintain reproducibility.
## CSE437: Data Science

Repository Name: cse437-airbnb-price-prediction-group-06

Section: 6

## Group Members

| Name | Student ID |
|---|---|
| Saptarshi Barman | 22301750 |
| Mahi Al Mahbub | 22101840 |

# Problem Statement

Airbnb listing prices are influenced by multiple factors, including property characteristics, location, host information, availability, and review-related features. Due to the complexity of these factors, accurately estimating an appropriate listing price manually can be challenging.

This project aims to develop a machine learning-based regression system to predict Airbnb listing prices using real-world London Airbnb data. The project also analyzes which features have the strongest relationship with listing prices and evaluates different machine learning models to identify the most accurate prediction approach.

# Dataset Information

## Dataset Name

Inside Airbnb London Detailed Listings Dataset

## Source

Inside Airbnb

## Dataset URL

https://insideairbnb.com/get-the-data/

## Description

This project uses the London Airbnb Detailed Listings dataset collected from Inside Airbnb.

The dataset contains real-world Airbnb listing information, including property details, host information, location attributes, availability information, review-related features, and pricing information.

The dataset was selected because it represents a real-world unclean dataset containing missing values, categorical variables, numerical features, and inconsistent data that require preprocessing before applying machine learning algorithms.

## Target Variable

The target variable for this project is:

log_price

The original Airbnb price value was transformed using logarithmic scaling to reduce skewness and improve machine learning model performance.

## Data Processing

The raw dataset was processed through the following steps:

- Missing value handling
- Feature transformation
- Categorical encoding
- Feature selection
- Leakage feature removal
- Train-test splitting

Processed datasets are stored inside:

data/processed/

Files include:

- X_train.csv
- X_test.csv
- y_train.csv
- y_test.csv
- feature_importance.csv

The original raw dataset is stored in:

data/raw/

The raw file remains unchanged to maintain reproducibility and preserve the original dataset.

# Research Questions

This project investigates the following three research questions:

1. Which property, location, host, and review-related features have the strongest relationship with Airbnb listing prices in London?

2. How do data preprocessing, feature selection, and dimensionality reduction affect the performance of Airbnb price prediction models?

3. Which machine-learning model provides the most accurate Airbnb price predictions after hyperparameter tuning, and what types of listings produce the largest prediction errors?

# Project Structure

cse437-airbnb-price-prediction-group-06/

├── README.md  
├── requirements.txt  
├── .gitignore  

├── data/  
│   ├── raw/  
│   ├── processed/  
│   └── README.md  

├── notebooks/  
│   ├── 01_data_audit_and_eda.ipynb  
│   ├── 02_preprocessing.ipynb  
│   ├── 03_feature_engineering.ipynb  
│   ├── 04_modeling_and_tuning.ipynb  
│   └── 05_evaluation_and_error_analysis.ipynb  

├── src/  
│   └── utils.py  

├── models/  

├── figures/  

└── report/  
    ├── report.md  
    └── report.pdf  


# How to Run the Project

## Install Required Libraries

Install dependencies using:

pip install -r requirements.txt


## Run Notebooks

Execute the notebooks in the following order:

1. notebooks/01_data_audit_and_eda.ipynb

2. notebooks/02_preprocessing.ipynb

3. notebooks/03_feature_engineering.ipynb

4. notebooks/04_modeling_and_tuning.ipynb

5. notebooks/05_evaluation_and_error_analysis.ipynb


All notebooks should be executed from the project root directory. The notebooks use relative paths and should be run in the given order to reproduce the complete workflow.

# Notebook Description

## 01_data_audit_and_eda.ipynb

Performs:

- Dataset inspection
- Exploratory data analysis
- Statistical analysis
- Data visualization


## 02_preprocessing.ipynb

Performs:

- Data cleaning
- Missing value handling
- Feature transformation
- Categorical encoding
- Train-test splitting


## 03_feature_engineering.ipynb

Performs:

- Feature analysis
- Feature selection
- Feature importance analysis
- Leakage feature removal
- Final feature preparation


## 04_modeling_and_tuning.ipynb

Performs:

- Model training
- Baseline model comparison
- Hyperparameter tuning using Grid Search
- Final model selection


## 05_evaluation_and_error_analysis.ipynb

Performs:

- Final model evaluation
- Prediction analysis
- Residual analysis
- Error investigation


# Final Model Performance

The final selected model is:

Tuned Gradient Boosting Regressor

Best Hyperparameters:

n_estimators = 200

learning_rate = 0.1

max_depth = 5


Final Performance:

| Metric | Score |
|---|---:|
| MAE | 0.169968 |
| RMSE | 0.268806 |
| R² Score | 0.881761 |


The model successfully predicts Airbnb listing prices and explains approximately 88% of the variation in price values.


# Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn


# Final Outcome

This project successfully developed a machine learning regression system for Airbnb price prediction.

The tuned Gradient Boosting Regressor achieved the best performance among the evaluated models. The results demonstrate that machine learning techniques can effectively capture complex relationships between Airbnb listing characteristics and pricing patterns.