import gym

import matplotlib.pyplot as plt
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions

env = gym.make('forex-v0', frame_bound=(50, 100), window_size=10)
observation = env.reset()

while True:
    action = env.action_space()
    _, reward, done, info = env.step(action)
    if done:
        print("info:", info)
        break

plt.cla()
env.render_all()
plt.show()
