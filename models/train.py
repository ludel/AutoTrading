import gym
import gym_anytrading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantstats as qs
from stable_baselines import A2C

df = gym_anytrading.datasets.STOCKS_GOOGL.copy()

window_size = 10
start_index = window_size
end_index = len(df)

env = gym.make('stocks-v0', df=df, window_size=window_size, frame_bound=(start_index, end_index))
print("> max_possible_profit:", env.max_possible_profit())

model = A2C('MlpLstmPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

observation = env.reset()
done = False
while not done:
    observation = observation[np.newaxis, ...]
    action, _states = model.predict(observation)
    observation, reward, done, info = env.step(action)

plt.cla()
env.render_all()
plt.show()

qs.extend_pandas()

net_worth = pd.Series(env.history['total_profit'], index=df.index[start_index + 1:end_index])
returns = net_worth.pct_change().iloc[1:]

qs.reports.metrics(returns)
