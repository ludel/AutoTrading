import yfinance as yf

data = yf.Ticker('kn.pa')
hist = data.history(period='2y', interval='1h')

# show actions (dividends, splits)
hist.to_csv('../data/banque/kn_1h_2d.csv')
