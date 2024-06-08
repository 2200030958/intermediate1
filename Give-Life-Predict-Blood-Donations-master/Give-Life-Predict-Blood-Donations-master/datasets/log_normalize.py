import numpy as np
import pandas as pd
from preprocess_data import create_target_column
from split_data import split_data

def log_normalize(X_train, X_test):
    X_train_log = X_train.copy()
    X_test_log = X_test.copy()
    for column in X_train_log.columns:
        X_train_log[column] = np.log1p(X_train_log[column])
        X_test_log[column] = np.log1p(X_test_log[column])
    return X_train_log, X_test_log

if __name__ == "__main__":
    df = pd.read_csv('transfusion.data')
    df = create_target_column(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_log, X_test_log = log_normalize(X_train, X_test)
    print(X_train_log.head())
