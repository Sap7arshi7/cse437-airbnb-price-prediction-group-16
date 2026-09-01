# Airbnb Price Prediction Using Machine Learning

## CSE437: Data Science Project Report

---

### Course Information

**Course:** CSE437 - Data Science  
**Section:** 6  
**Semester:** Summer 2026  

---

### Group Members

| Name | Student ID |
|---|---|
| Saptarshi Barman | 22301750 |
| Mahi Al Mahbub | 22101840 |

---

### GitHub Repository

Repository Link:

https://github.com/Sap7arshi7/cse437-airbnb-price-prediction-group-16

---

### Submission Date

3 September 2026

---

# Summary

Airbnb has become one of the largest online accommodation platforms, where accurate pricing plays an important role in attracting customers and maximizing revenue. However, Airbnb listing prices depend on multiple interacting factors such as property characteristics, location, host information, availability, and review-related features. Due to these complex relationships, manually estimating suitable prices can be challenging.

This project develops a machine learning-based regression system for predicting Airbnb listing prices using the London Airbnb Detailed Listings Dataset collected from Inside Airbnb. The project follows a complete data science workflow including data auditing, exploratory data analysis, preprocessing, feature engineering, model development, hyperparameter tuning, and error analysis.

Several regression models were evaluated, including Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor. After baseline comparison, Gradient Boosting was selected for hyperparameter optimization using Grid Search. The tuned Gradient Boosting model achieved the best performance with an MAE of 0.169968, RMSE of 0.268806, and R² score of 0.881761.

The findings demonstrate that machine learning models can effectively capture complex relationships between Airbnb listing characteristics and pricing patterns.

---

# 1. Problem and Dataset

## 1.1 Problem Statement

Airbnb listing prices are influenced by multiple factors, including property characteristics, location, host information, availability, and review-related features. Due to the complexity of these factors, accurately estimating an appropriate listing price manually can be difficult.

This project aims to develop a machine learning-based regression system to predict Airbnb listing prices using real-world London Airbnb data. The project also investigates which features have the strongest relationship with listing prices and evaluates different machine learning models to identify the most accurate prediction approach.

The problem is formulated as a supervised regression task where the model learns the relationship between listing features and the corresponding price value.

---

## 1.2 Dataset

### Dataset Name

Inside Airbnb London Detailed Listings Dataset

### Source

Inside Airbnb

### Dataset URL

https://insideairbnb.com/get-the-data/

### Dataset Description

This project uses the London Airbnb Detailed Listings Dataset collected from Inside Airbnb.

The dataset contains real-world Airbnb listing information including:

- Property details
- Host information
- Location attributes
- Availability information
- Review-related features
- Pricing information

The dataset was selected because it represents a real-world unclean dataset containing missing values, numerical variables, categorical variables, and inconsistent data requiring preprocessing before applying machine learning algorithms.

### Dataset Size

The dataset contains a large collection of Airbnb listing records with multiple property, host, location, availability, review, and pricing-related attributes. This provides sufficient data for preprocessing, feature analysis, machine learning model training, and evaluation.

The raw dataset is stored inside:
data/raw/

The processed datasets generated after preprocessing are stored inside:
data/processed/

---

## 1.3 Target Variable

The target variable used in this project is:
log_price

The original Airbnb price values were transformed using logarithmic scaling.

The logarithmic transformation was applied because price data usually contains high-value outliers and a skewed distribution. Transforming the target variable reduces the impact of extreme values and improves model learning performance.

The final task is therefore a regression problem where machine learning models predict the transformed listing price.

---

## 1.4 Research Questions

This project investigates the following three research questions:

### Research Question 1

Which property, location, host, and review-related features have the strongest relationship with Airbnb listing prices in London?

### Research Question 2

How do data preprocessing and feature selection affect the performance of Airbnb price prediction models?

### Research Question 3

Which machine-learning model provides the most accurate Airbnb price predictions after hyperparameter tuning, and what types of listings produce the largest prediction errors?

---

# 2. Data Handling and Preprocessing

## 2.1 Data Quality Audit

The first stage of the project involved analyzing the raw dataset to understand its structure and quality.

The data audit included:

- Checking dataset dimensions
- Identifying numerical and categorical features
- Inspecting missing values
- Understanding feature distributions
- Detecting possible inconsistencies

This step helped determine the required preprocessing operations before model training.

![Missing Values Analysis](../figures/missing_values.png)

**Figure 1: Missing Value Analysis**

The missing value analysis helped identify incomplete attributes in the original dataset and guided the preprocessing decisions applied during data cleaning.

---

## 2.2 Missing Value Handling

The dataset contained missing values in several attributes. These missing values were handled during preprocessing to ensure that machine learning algorithms could process the data correctly.

The preprocessing pipeline included:

- Identifying missing values
- Applying suitable handling techniques
- Preparing clean input features

After preprocessing, the processed dataset contained no unresolved missing values affecting model training.

---

## 2.3 Data Cleaning and Transformation

Several preprocessing operations were applied:

- Removal of unnecessary attributes
- Handling inconsistent values
- Encoding categorical variables
- Transforming numerical features
- Preparing final input features

Categorical variables were converted into numerical representations so that machine learning algorithms could process them.

---

## 2.4 Train-Test Split

The processed dataset was divided into training and testing subsets.

The training data was used for:

- Model learning
- Feature analysis
- Hyperparameter tuning

The testing data was kept separate for final evaluation.

Processed files generated:
X_train.csv
X_test.csv
y_train.csv
y_test.csv

---

## 2.5 Before and After Preprocessing

The preprocessing stage transformed the raw Airbnb listing data into a machine-learning-ready dataset.

| Processing Step | Before | After |
|---|---|---|
| Missing values | Present in raw dataset | Handled during preprocessing |
| Categorical features | Text-based categories | Numerically encoded |
| Unnecessary features | Included in raw dataset | Removed |
| Feature leakage risk | Possible during raw preparation | Leakage-prone features removed |
| Dataset format | Raw listing information | Processed dataset ready for model training |

---

# 3. Statistical Analysis

## 3.1 Descriptive Statistics

Statistical analysis was performed to understand the distribution and characteristics of the dataset.

The analysis included:

- Mean values
- Standard deviation
- Minimum and maximum values
- Feature distributions

These statistical observations helped identify important patterns within Airbnb listing data.

![Raw Price Distribution](../figures/raw_price_distribution.png)

**Figure 2: Airbnb Price Distribution**

The price distribution shows the original variation of Airbnb listing prices and explains the need for logarithmic transformation of the target variable.

---

## 3.2 Feature Relationship Analysis

Correlation analysis was performed to investigate relationships between numerical features.

The analysis focused on identifying relationships between:

- Property characteristics
- Location-related variables
- Availability information
- Review-related attributes
- Price-related features

Feature relationships provided an initial understanding of which variables may contribute significantly to Airbnb pricing.

![Feature Correlation Heatmap](../figures/correlation_heatmap.png)

**Figure 3: Feature Correlation Heatmap**

The correlation heatmap shows relationships between numerical features and helps identify possible factors influencing Airbnb listing prices.

![Price by Room Type](../figures/price_by_room_type.png)

**Figure 4: Price Variation by Room Type**

This visualization shows how different property types influence Airbnb listing prices.

---

## 3.3 Statistical Observations

The analysis showed that Airbnb prices are influenced by multiple factors rather than a single variable.

Important observations include:

- Property characteristics strongly affect pricing.
- Location-related information contributes significantly to price variation.
- Host and review-related information provides additional predictive value.
- Complex relationships exist between multiple features, making machine learning suitable for this task.

---
# 4. Feature Engineering

## 4.1 Feature Preparation

Feature engineering was performed to improve the quality of the input data and prepare the dataset for machine learning models.

The feature engineering process included:

- Reviewing feature importance
- Removing unnecessary attributes
- Selecting relevant variables
- Preparing numerical and encoded categorical features

The objective was to create a compact and informative feature set while reducing unnecessary complexity.

---

## 4.2 Feature Selection

Feature selection was performed to identify the most useful features contributing to Airbnb price prediction.

The Gradient Boosting model was used to analyze feature importance and determine which variables had higher predictive contribution.

Feature selection helps:

- Reduce unnecessary information
- Improve model efficiency
- Reduce the possibility of overfitting
- Improve model interpretation

![Feature Importance](../figures/feature_importance_final.png)

**Figure 5: Feature Importance Analysis**

The feature importance visualization shows the contribution of different features toward Airbnb price prediction. This analysis helps identify the variables with higher predictive influence.

---

## 4.3 Leakage Prevention

During feature preparation, features that could introduce unrealistic prediction advantages were reviewed and removed.

Preventing data leakage ensures that the model learns genuine relationships between listing characteristics and prices instead of relying on information that would not be available during real-world prediction.

---

## 4.4 Final Feature Set

After preprocessing and feature engineering, the final dataset consisted of processed numerical and encoded categorical features.

The final processed feature matrices were:
X_train.csv
X_test.csv

The final feature set was used for all machine learning models and evaluation experiments.

---

# 5. Modeling and Validation

## 5.1 Validation Strategy

The project used a supervised regression approach.

The dataset was divided into training and testing subsets. The training data was used for model development, while the testing data was reserved for final performance evaluation.

The validation strategy ensured that the final model performance was measured on unseen data.

---

## 5.2 Baseline Models

Three machine learning regression models were evaluated:

## Linear Regression

Linear Regression was used as the baseline model to understand the relationship between input features and Airbnb prices.

It provides a simple reference point for comparing more complex algorithms.

---

## Random Forest Regressor

Random Forest is an ensemble learning method that combines multiple decision trees.

It can capture nonlinear relationships between Airbnb listing features and price values.

---

## Gradient Boosting Regressor

Gradient Boosting is an ensemble learning technique that builds multiple weak learners sequentially to improve prediction accuracy.

This model was selected for further optimization because of its ability to handle complex feature relationships.

---

## 5.3 Evaluation Metrics

The following regression metrics were used:

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values.

Lower MAE indicates better prediction accuracy.

### Root Mean Squared Error (RMSE)

RMSE measures prediction error while giving higher importance to larger errors.

Lower RMSE indicates improved model performance.

### R² Score

R² measures how much variation in the target variable is explained by the model.

A higher R² value indicates better predictive capability.

---

## 5.4 Baseline Model Comparison

The baseline model performance is shown below:

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 0.2927 | 0.3961 | 0.7432 |
| Random Forest Regressor | 0.2107 | 0.3128 | 0.8399 |
| Gradient Boosting Regressor | 0.2511 | 0.3481 | 0.8017 |

The comparison shows that Random Forest provided strong baseline performance. However, Gradient Boosting was selected for further optimization using hyperparameter tuning.

---

# 6. Hyperparameter Tuning

## 6.1 Tuning Method

Hyperparameter tuning was performed using Grid Search Cross Validation.

The objective was to identify the best combination of parameters for the Gradient Boosting model.

Grid Search evaluates multiple parameter combinations and selects the configuration that provides the best validation performance.

---

## 6.2 Search Space

The following parameters were optimized:

| Parameter | Values |
|---|---|
| n_estimators | 100, 200 |
| learning_rate | 0.05, 0.1 |
| max_depth | 3, 5 |

---

## 6.3 Best Parameters

The optimized Gradient Boosting model selected the following parameters:
n_estimators = 200

learning_rate = 0.1

max_depth = 5

These parameters produced the best validation performance among the tested combinations.

---

## 6.4 Tuned Model Performance

The tuned Gradient Boosting model achieved:

| Metric | Score |
|---|---:|
| MAE | 0.169968 |
| RMSE | 0.268806 |
| R² Score | 0.881761 |

The tuned model significantly improved prediction accuracy compared to the baseline models.

---

# 7. Results, Visualization and Error Analysis

## 7.1 Final Test Performance

The final selected model was the Tuned Gradient Boosting Regressor.

The model achieved an R² score of 0.881761, meaning that approximately 88% of the variation in Airbnb prices was explained by the model.

The final results demonstrate that machine learning models can effectively predict Airbnb listing prices from real-world listing information.

---

## 7.2 Prediction Visualization

The actual versus predicted visualization was used to evaluate how closely the model predictions matched the real Airbnb prices.

![Actual vs Predicted](../figures/actual_vs_predicted.png)

**Figure 6: Actual vs Predicted Price Values**

The plot shows that most predictions follow the general trend of actual values, indicating that the tuned Gradient Boosting model successfully learned the relationship between input features and price values.

---

## 7.3 Residual Analysis

Residual analysis was performed to understand prediction errors.

The residual distribution showed that most errors were centered around zero, indicating that the model did not show significant systematic bias.

The largest errors mainly occurred for listings with extreme price values because these cases are less common and more difficult for the model to learn.

![Residual Distribution](../figures/residual_distribution.png)

**Figure 7: Residual Distribution**

The residual distribution shows the difference between actual and predicted values. Most residuals are concentrated around zero, indicating limited prediction bias.

![Residual Error Analysis](../figures/residual_error_analysis.png)

**Figure 8: Residual Error Analysis**

The error analysis highlights cases where the model produces larger prediction errors. These errors are mainly associated with unusual or extreme-price listings.

---

## 7.4 Answers to Research Questions

## Research Question 1

**Which property, location, host, and review-related features have the strongest relationship with Airbnb listing prices in London?**

The analysis showed that Airbnb prices are affected by multiple feature groups including property characteristics, location information, host-related variables, and review-related attributes. Feature importance analysis helped identify the variables contributing most strongly to price prediction.

---

## Research Question 2

**How do data preprocessing and feature selection affect the performance of Airbnb price prediction models?**

Data preprocessing improved model performance by cleaning the dataset, handling missing values, encoding categorical variables, and preparing consistent input features. Feature selection reduced unnecessary information and helped the models focus on the most relevant attributes, improving efficiency and prediction quality.

---

## Research Question 3

**Which machine-learning model provides the most accurate Airbnb price predictions after hyperparameter tuning, and what types of listings produce the largest prediction errors?**

The tuned Gradient Boosting Regressor provided the most accurate prediction performance with an R² score of 0.881761. The largest prediction errors occurred mainly for extreme-price listings where fewer training examples were available.

---

# 8. Limitations and Next Steps

Although the developed system achieved strong prediction performance, several limitations remain.

First, the dataset represents Airbnb listings from London only, meaning the model may not directly generalize to other cities or markets.

Second, external factors such as seasonal demand, special events, economic changes, and market trends were not included in the dataset.

Third, extreme luxury listings may still produce larger prediction errors because these cases have fewer similar examples in the training data.

Future improvements may include:

- Using larger datasets from multiple locations
- Including temporal pricing information
- Adding external market factors
- Exploring advanced ensemble learning approaches

---

# 9. Contributions

| Member | Student ID | Contribution |
|---|---|---|
| Saptarshi Barman | 22301750 | Data preprocessing, feature engineering, model development, hyperparameter tuning, and evaluation analysis |
| Mahi Al Mahbub | 22101840 | Exploratory data analysis, visualization, documentation, report preparation, and repository organization |

---

# Conclusion

This project developed a machine learning-based Airbnb price prediction system using real-world London Airbnb listing data. The complete workflow included data auditing, preprocessing, feature engineering, model comparison, hyperparameter tuning, and error analysis.

Three regression models were evaluated, and the tuned Gradient Boosting Regressor achieved the best performance with an MAE of 0.169968, RMSE of 0.268806, and R² score of 0.881761.

The results demonstrate that machine learning models can effectively capture complex relationships between Airbnb listing characteristics and pricing patterns. Future improvements can focus on incorporating additional market information, larger datasets, and advanced modeling techniques to further improve prediction accuracy.

---

# References

1. Inside Airbnb. Airbnb Dataset.  
https://insideairbnb.com/get-the-data/

2. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research.

3. McKinney, W. (2010). Data Structures for Statistical Computing in Python.

4. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.