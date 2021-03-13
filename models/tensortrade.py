import pandas as pd
import ta
import tensortrade.env.default as default
from stable_baselines3 import A2C, DQN, PPO
from tensortrade.env.default.actions import BSH
from tensortrade.env.default.rewards import RiskAdjustedReturns
from tensortrade.feed.core import Stream, DataFeed
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import EUR, Instrument
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio

symbol = 'ALO'
ink = 'alstom'
inst = Instrument(symbol, 1, 'Mcphy energy stock')

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

cash = Wallet(exchange, 3_000 * EUR)
asset = Wallet(exchange, 0 * inst)

portfolio = Portfolio(EUR, [
    cash,
    asset
])

renderer_feed = DataFeed([
    Stream.source(list(data["ALO:date"])).rename("date"),
    Stream.source(list(data["ALO:open"]), dtype="float").rename("open"),
    Stream.source(list(data["ALO:high"]), dtype="float").rename("high"),
    Stream.source(list(data["ALO:low"]), dtype="float").rename("low"),
    Stream.source(list(data["ALO:close"]), dtype="float").rename("close"),
    Stream.source(list(data["ALO:volume"]), dtype="float").rename("volume")
])
reward_scheme = RiskAdjustedReturns()
action_scheme = BSH(cash, asset)

env = default.create(
    portfolio=portfolio,
    action_scheme=action_scheme,
    reward_scheme=reward_scheme,
    feed=feed,
    renderer_feed=renderer_feed,
    renderer=default.renderers.PlotlyTradingChart(display=False, save_format='html', path='../chart'),
    window_size=20
)

# agent = DQN(env)

agent = PPO('MlpPolicy', env, verbose=0)
agent.learn(total_timesteps=50000)
# agent.train(n_steps=300, n_episodes=10, render_interval=999999, cyclic_steps=True)


# observation = env.reset()
print('Net worth {}'.format(portfolio.net_worth))
observation = env.reset()
while True:
    action, _ = agent.predict(observation)
    observation, reward, done, info = env.step(action)
    if done:
        break
env.render()
env.save()
