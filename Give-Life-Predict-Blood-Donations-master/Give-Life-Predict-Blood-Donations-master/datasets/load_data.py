# load_data.py
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

if __name__ == "__main__":
    df = load_data('transfusion.data')
    print(df.head())

# load_data.py (continued)
def inspect_data(df):
    print(df.info())
    print(df.head())

if __name__ == "__main__":
    df = load_data('transfusion.data')
    inspect_data(df)
