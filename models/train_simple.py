import itertools

import pandas as pd
import ta
import tensortrade.env.default as default
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import EUR, Instrument
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio

from agent.A2C import A2C

MCPHY = Instrument('MCPHY', 2, 'Mcphy energy stock')
symbol = 'MCPHY'

data = pd.read_csv('../data/mcphy_15m_1mo.csv').drop('Datetime', axis=1)
data = ta.add_all_ta_features(data, 'Open', 'High', 'Low', 'Close', 'Volume', fillna=True)
data.columns = [symbol + ":" + name.lower() for name in data.columns]

data_feed = DataFeed([
    Stream.source(list(data[col]), dtype="float") for col in data.columns
])
exchange = Exchange("mcphy", service=execute_order)(
    Stream.source(list(data['MCPHY:close'])).rename("EUR-MCPHY")
)
renderer_feed = DataFeed([
    Stream.source(data[c].tolist(), dtype="float").rename(c) for c in data]
)

best_agent_score = 0
best_agent = None
for n_steps, n_episodes in itertools.combinations([2, 5, 10, 20, 50, 100], 2):
    portfolio = Portfolio(EUR, [
        Wallet(exchange, 3_000 * EUR),
        Wallet(exchange, 0 * MCPHY)
    ])
    env = default.create(
        portfolio=portfolio,
        action_scheme='managed-risk',
        reward_scheme='risk-adjusted',
        feed=data_feed,
        renderer=default.renderers.PlotlyTradingChart(),
        window_size=20
    )
    env.reset()

    agent = A2C(env)
    agent.train(n_steps=n_steps, n_episodes=n_episodes)
    print('Net worth {}'.format(portfolio.net_worth))
    if portfolio.net_worth > best_agent_score:
        best_agent_score = portfolio.net_worth
        best_agent = agent
        print('New best agent with {}'.format(best_agent_score))
