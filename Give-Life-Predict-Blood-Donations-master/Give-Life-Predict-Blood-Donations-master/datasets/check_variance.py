import pandas as pd
from preprocess_data import create_target_column
from split_data import split_data

def check_variance(X_train):
    print(X_train.var())

if __name__ == "__main__":
    df = pd.read_csv('transfusion.data')
    df = create_target_column(df)
    X_train, X_test, y_train, y_test = split_data(df)
    check_variance(X_train)
