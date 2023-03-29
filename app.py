import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#Auto importing all the packages using pyforest 
#import pyforest 


def load_dataset(file_path):
    """Loads the dataset from the specified file path and returns a pandas DataFrame."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)


def explore_dataset(data):
    """Provides summary statistics and visualizations of the dataset."""
    print("Dataset shape:", data.shape)
    print("Dataset data types:\n", data.dtypes)
    print("Number of missing values:\n", data.isna().sum())

    # Visualize the distribution of values for each column
    for col in data.columns:
        if data[col].dtype == 'object':
            sns.countplot(data[col])
            plt.title(col)
            plt.show()
        else:
            sns.histplot(data[col])
            plt.title(col)
            plt.show()


def clean_dataset(data, drop_missing=True):
    """Performs data cleaning on the dataset."""
    # Remove duplicate rows
    data.drop_duplicates(inplace=True)

    # Fill missing values
    data.fillna(method='ffill', inplace=True)

    # Drop columns with a high percentage of missing values
    missing_percent = data.isnull().sum() / len(data) * 100
    to_drop = list(missing_percent[missing_percent > 50].index)
    data.drop(to_drop, axis=1, inplace=True)

    # Drop rows with missing values if drop_missing is True
    if drop_missing:
        data.dropna(inplace=True)
        
    return data


#Signal that the code is working without errors.(The following code should be commented out) 
#print('The package is working without errors.')