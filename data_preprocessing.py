import pandas as pd

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df.rename(columns={'Company ': 'Company', 'Market Cap(in B USD)': 'Market_cap',
                       'Inflation Rate(in US)': 'Inflation_rate', 'Category': 'Industry'}, inplace=True)
    df['Industry'] = df['Industry'].replace('BANK', 'Bank')
    df.columns = df.columns.str.replace(' ', '_')

    df['Market_cap'].fillna(value=df['Market_cap'].mean(), inplace=True)
    return df
