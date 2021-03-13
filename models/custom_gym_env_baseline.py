import os

import pandas as pd
import ta
from stable_baselines import DQN

from env.trading import TradingEnv

MAX_STEP = 200


def read_csv(file):
    df = pd.read_csv(file).drop('Date', axis=1)
    df = df[df['Open'] != 0]
    return ta.add_all_ta_features(df, 'Open', 'High', 'Low', 'Close', 'Volume', fillna=True)


def get_banque_data():
    for file_name in os.listdir('../data/banque'):
        yield read_csv(f'../data/banque/{file_name}')


#data = list(get_banque_data())
data = [read_csv('../data/banque/bnp_1h_2d.csv')]

env = TradingEnv(data, initial_account=3000, max_step=MAX_STEP, window_size=20)
# env = DummyVecEnv([lambda: env])

agent = DQN('MlpPolicy', env, verbose=0, learning_rate=0.0001)
# agent.load('../save/save.zip')
agent.learn(total_timesteps=100_000)


def test(max_step):
    env.max_step = max_step
    observation = env.reset()
    while True:
        action, _ = agent.predict(observation)
        observation, reward, done, info = env.step(action)
        # env.render()
        if done:
            break

    print('==> Total env reward ', env.total_reward)
    print('==> Total env net worth', env.net_worth)


test(200)
env.plot()
