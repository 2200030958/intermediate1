import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df):
    X = df.drop(columns='Target')
    y = df['Target']
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

if __name__ == "__main__":
    df = pd.read_csv('transfusion.data')
    from preprocess_data import create_target_column  # Ensure correct import
    df = create_target_column(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
