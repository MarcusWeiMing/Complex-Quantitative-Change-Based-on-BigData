
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    data = data.drop_duplicates()
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data

def normalize_data(data):
    scaler = StandardScaler()
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data

def detect_outliers(data):
    from scipy.stats import zscore
    z_scores = np.abs(zscore(data.select_dtypes(include=[np.number])))
    data = data[(z_scores < 3).all(axis=1)]
    return data
