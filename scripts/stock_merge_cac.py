import pandas as pd

df_stock = pd.read_csv('../data/stocks_cac_40.csv')
df = pd.DataFrame({'date': df_stock['date'].unique()})
df = df.sort_values('date')

for symbol in df_stock['symbol'].unique():
    company_data = df_stock.loc[df_stock['symbol'] == symbol]
    company_data = company_data.rename(columns={'price': symbol})
    df = pd.merge(df, company_data[['date', symbol]], on='date', how='left')

df.to_csv('../data/stocks_cac_40_merged.csv', index=False)
