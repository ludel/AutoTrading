import pandas as pd
import ta
from stable_baselines3 import PPO, A2C
import os
from env.trading import TradingEnv

data = []
for file_name in os.listdir('../data/banque'):
    df = pd.read_csv(f'../data/banque/{file_name}').drop('Date', axis=1)
    df = df[df['Open'] != 0]
    df = ta.add_all_ta_features(df, 'Open', 'High', 'Low', 'Close', 'Volume', fillna=True)
    data.append(df)

env = TradingEnv(data, initial_account=3000, max_step=200, window_size=20)

agent = PPO('MlpPolicy', env, verbose=0, n_steps=200, learning_rate=0.0001)
# agent.load('../save/save.zip')
agent.learn(total_timesteps=50_000)


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
