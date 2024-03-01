import numpy as np
import pandas as pd
import seaborn
from column_cleaner import clean_numeric_column

file_path = "avg_salary_it_2022.csv"

# load file using pandas
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

# info de la target_column, min, max, mean, std,5 0%, etc.
print(dataset[TARGET_COLUMN].describe())
