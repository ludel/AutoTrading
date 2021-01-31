import pandas as pd
import ta
import tensortrade.env.default as default
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import EUR, Instrument
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio

from agent.DQN import DQN

symbol = 'ALO'
ink = 'alstom'
inst = Instrument(symbol, 2, 'Mcphy energy stock')

data = pd.read_csv('../data/alstom_1h_2y.csv')
data = ta.add_all_ta_features(data, 'Open', 'High', 'Low', 'Close', 'Volume', fillna=True)
data.columns = [symbol + ":" + name.lower() for name in data.columns]

feed = DataFeed([
    Stream.source(list(data[col]), dtype="float") for col in data.columns if col != f'{symbol}:date'
])
feed.compile()

exchange = Exchange(ink, service=execute_order)(
    Stream.source(list(data[f'{symbol}:close'])).rename(f"EUR-{symbol}")
)

portfolio = Portfolio(EUR, [
    Wallet(exchange, 3_000 * EUR),
    Wallet(exchange, 0 * inst)
])

renderer_feed = DataFeed([
    Stream.source(list(data["ALO:date"])).rename("date"),
    Stream.source(list(data["ALO:open"]), dtype="float").rename("open"),
    Stream.source(list(data["ALO:high"]), dtype="float").rename("high"),
    Stream.source(list(data["ALO:low"]), dtype="float").rename("low"),
    Stream.source(list(data["ALO:close"]), dtype="float").rename("close"),
    Stream.source(list(data["ALO:volume"]), dtype="float").rename("volume")
])

env = default.create(
    portfolio=portfolio,
    action_scheme="managed-risk",
    reward_scheme="risk-adjusted",
    feed=feed,
    renderer_feed=renderer_feed,
    window_size=20
)
env.reset()

agent = DQN(env)
agent.train(n_steps=200, n_episodes=3, render_interval=200000)
print('Net worth {}'.format(portfolio.net_worth))

observation = env.reset()

while True:
    action = agent.get_action(observation)
    observation, reward, done, info = env.step(action)
    if done:
        break

env.render()
env.save()
