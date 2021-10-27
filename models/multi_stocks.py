import pandas as pd
from stable_baselines3 import PPO

from env.trading_v2 import TradingEnvV2

# Preprocessing
df = pd.read_csv('../data/cac_merge_clean.csv', index_col=0)

# Env
env = TradingEnvV2(df, initial_account=50_000)
self = env  # console feature
nb_actions = env.action_space.shape[0]

print('=> Initial Model')
model = PPO('MlpPolicy', env)

print('=> Learn ...')
model.learn(total_timesteps=10_000)

print('=> Test')
env.df = env.df.loc[env.df.index > 4000]
obs = env.reset()
done = False
i = 0
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    i += 1
    if i > 500:
        break

env.plot()
