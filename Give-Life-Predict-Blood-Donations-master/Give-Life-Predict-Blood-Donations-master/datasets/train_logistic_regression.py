from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def train_logistic_regression(X_train_log, y_train, X_test_log, y_test):
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_log, y_train)
    y_pred = log_reg.predict(X_test_log)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, conf_matrix

if __name__ == "__main__":
    import pandas as pd
    from preprocess_data import create_target_column
    from split_data import split_data
    from log_normalize import log_normalize
    
    df = pd.read_csv('transfusion.data')
    df = create_target_column(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_log, X_test_log = log_normalize(X_train, X_test)
    accuracy, conf_matrix = train_logistic_regression(X_train_log, y_train, X_test_log, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{conf_matrix}')
