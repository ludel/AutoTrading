import pandas as pd
import ta
import tensortrade.env.default as default
from tensortrade.agents import DQNAgent
from tensortrade.env.default.actions import BSH
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import EUR, Instrument
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio

from agent.A2C import A2C

symbol = 'ALO'
ink = 'alstom'
inst = Instrument(symbol, 2, 'Mcphy energy stock')


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

portfolio = Portfolio(EUR, [
    Wallet(exchange, 3_000 * EUR),
    Wallet(exchange, 0 * MCPHY)
])

action_scheme = default.actions.BSH(cash=Wallet(exchange, 3_000 * EUR), asset=Wallet(exchange, 0 * MCPHY))

env = default.create(
    portfolio=portfolio,
    action_scheme='simple',
    reward_scheme='simple',
    feed=data_feed,
    renderer_feed=data_feed,
    renderer=default.renderers.PlotlyTradingChart(),
    window_size=20
)
env.reset()

agent = DQNAgent(env)
agent.train(n_steps=200, n_episodes=5)
# agent.restore(path='../save/', actor_filename='actor_network.hdf5', critic_filename='critic_network.hdf5')

observation = env.reset()
while True:
    action = agent.get_action(observation)
    observation, reward, done, info = env.step(action)
    if done:
        print("info:", info)
        break
