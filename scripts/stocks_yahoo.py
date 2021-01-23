import yfinance as yf
import pandas as pd


df_companies = pd.read_csv('data/companies_cac_40.csv')

for code in df_companies['symbol'].values:
    data = yf.Ticker('{}.PA'.format(code))
    hist = data.history(period="max")

# show actions (dividends, splits)
print(msft)

