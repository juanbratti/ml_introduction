import numpy as np
import pandas as pd
import seaborn
from src.data.column_cleaner import clean_numeric_column
from sklearn.preprocessing import OneHotEncoder
import os

file_path = "/home/juanbratti/Desktop/ml_introduction/data/avg_salary_it_2022.csv"

raw_dataset = pd.read_csv(file_path)

# data cleaning & target, features identification
TARGET_COLUMN = 'net_monthly_salary'

COLUMN_RENAMES = {
    'Dónde estás trabajando': 'province',
    'Dedicación': 'dedication',
    'Tipo de contrato': 'contract_type',
    'Último salario mensual o retiro NETO (en tu moneda local)': TARGET_COLUMN,
    'Tengo (edad)': 'age',
    'Me identifico (género)': 'gender'
}

FEATURE_COLUMNS = list(COLUMN_RENAMES.keys())

dataset = raw_dataset[FEATURE_COLUMNS]
dataset = dataset.rename(columns=COLUMN_RENAMES)

# salary is a string, should be int, and there are NaN values
dataset = clean_numeric_column(TARGET_COLUMN, dataset)
dataset = clean_numeric_column('age', dataset)

# remove rows with outlier values
dataset = dataset[(dataset[TARGET_COLUMN] > 50000) & (dataset[TARGET_COLUMN] < 2000000)]
dataset = dataset[(dataset['age'] > 18) & (dataset['age'] < 99)]

# target_column, min, max, mean, std,5 0%, etc.
# print(dataset[TARGET_COLUMN].describe())

# characteristics we need to encode
CATEGORICAL_FEATURES = ['province', 'dedication', 'contract_type', 'gender']
NUMERICAL_FEATURES = ['age'] # already numerical characteristics

# Obtain only categoric data from dataframe
categorical_data = dataset[CATEGORICAL_FEATURES]

# create one-hot enc instance
encoder = OneHotEncoder()
# adjust & transform categoric data
categorical_encoded=encoder.fit_transform(categorical_data).toarray()
# obtain encoded categoric data as dataframe
categorical_encoded = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES))
numerical_data=dataset[NUMERICAL_FEATURES]
# concatenate encoded data and numeric data
encoded_dataset = pd.concat(
    [categorical_encoded, numerical_data.reset_index(drop=True)],
    axis=1)

assert TARGET_COLUMN not in encoded_dataset # we cannot use target column to train the model!!!

encoded_dataset.to_csv("/home/juanbratti/Desktop/ml_introduction/data/encoded_dataset.csv", index=False)

