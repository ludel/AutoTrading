from env.trading import TradingEnv
from preprocessing.feature import read_csv

data = [read_csv('../data/banque/bnp_1h_2d.csv')]
reward_function = ['sharpe_ratio', 'calmar_ratio', 'sortino_ratio', 'omega_ratio']

env = TradingEnv(data, initial_account=3000, window_size=20, reward_method='sharpe_ration')

nb_actions = env.action_space.n
i = 0
total_reward = 0
env.reset()

while True:
    i += 1
    action = env.action_space.sample()
    _, reward, done, info = env.step(action)
    total_reward += reward
    if done or i > 200:
        break
env.plot()

print('==> Total env reward ', total_reward)
