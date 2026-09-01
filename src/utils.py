import pandas as pd
import numpy as np


def load_dataset(path):
    """
    Load dataset from CSV file.
    """
    return pd.read_csv(path)


def calculate_missing_values(df):
    """
    Returns missing value statistics.
    """
    missing = df.isnull().sum()
    percentage = (missing / len(df)) * 100
    
    return pd.DataFrame({
        "Missing Values": missing,
        "Percentage": percentage
    })


def remove_outliers(df, column):
    """
    Remove extreme values using IQR method.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[column] >= lower) & (df[column] <= upper)]