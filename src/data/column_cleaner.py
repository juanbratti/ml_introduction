import numpy as np
import pandas as pd

def clean_numeric_column(colname:str, dataset):
    # convert column to numeric
    dataset[colname] = pd.to_numeric(dataset[colname], errors='coerce')
    # drop rows with NaN values
    dataset.dropna(subset=[colname], inplace=True)
    # convert column to int
    dataset[colname] = dataset[colname].astype(int)
    return dataset