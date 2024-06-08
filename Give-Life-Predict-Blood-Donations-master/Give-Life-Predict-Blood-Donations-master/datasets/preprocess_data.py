import pandas as pd

def create_target_column(df):
    df.columns = ['Recency', 'Frequency', 'Monetary', 'Time', 'Donated_Mar_2007']
    df['Target'] = df['Donated_Mar_2007']
    df.drop(columns='Donated_Mar_2007', inplace=True)
    return df

def check_target_incidence(df):
    print(df['Target'].value_counts(normalize=True))

if __name__ == "__main__":
    df = pd.read_csv('transfusion.data')
    df = create_target_column(df)
    check_target_incidence(df)
