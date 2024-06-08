import pandas as pd
from split_data import split_data
from preprocess_data import create_target_column
from log_normalize import log_normalize
from train_tpot import train_tpot
from train_logistic_regression import train_logistic_regression

def load_data(file_path):
    return pd.read_csv(file_path)

def main():
    df = load_data('transfusion.data')
    df = create_target_column(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_log, X_test_log = log_normalize(X_train, X_test)
    tpot_model = train_tpot(X_train, y_train)
    accuracy, conf_matrix = train_logistic_regression(X_train_log, y_train, X_test_log, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{conf_matrix}')

if __name__ == "__main__":
    main()
