import pandas as pd
from preprocess_data import create_target_column
from split_data import split_data
from tpot import TPOTClassifier

def train_tpot(X_train, y_train):
    tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20, random_state=42, config_dict='TPOT sparse')
    tpot.fit(X_train, y_train)
    return tpot

if __name__ == "__main__":
    df = pd.read_csv('transfusion.data')
    df = create_target_column(df)
    X_train, X_test, y_train, y_test = split_data(df)
    tpot_model = train_tpot(X_train, y_train)
