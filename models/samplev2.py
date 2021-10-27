import pandas as pd

from env.trading_v2 import TradingEnvV2

data = pd.read_csv('../data/cac_merge_clean.csv', index_col=0).tail(30_000)

env = TradingEnvV2(data, initial_account=10_000)

total_reward = 0
env.reset()

for i in range(300):
    action = env.action_space.sample()
    _, reward, done, info = env.step(action)
    if done:
        break

print('==> Total env reward ', total_reward)
