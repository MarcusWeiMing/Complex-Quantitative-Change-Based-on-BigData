
from sklearn.linear_model import LinearRegression
import numpy as np

def train_time_series_model(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions
