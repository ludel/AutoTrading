import pandas as pd
import yfinance as yf

from config import cac_tickers
from preprocessing.feature import add_feature

df = pd.DataFrame()

for ticket in cac_tickers:
    print(ticket)
    data = yf.Ticker(ticket)
    hist = data.history(period='max', interval='1d', start='2005-01-01')
    hist = add_feature(hist)
    hist['Code'] = ticket
    df = df.append(hist)

df = df.sort_values('Date')
df['Day'] = df.groupby('Date').ngroup()
df.to_csv('../cac_merge.csv')
