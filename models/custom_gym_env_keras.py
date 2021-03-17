import os

import pandas as pd
import ta
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy

from env.trading import TradingEnv

MAX_STEP = 300


def read_csv(file):
    df = pd.read_csv(file).drop('Date', axis=1)
    df = df[df['Open'] != 0]
    df = ta.add_volume_ta(df, 'High', 'Low', 'Close', 'Volume', fillna=True)
    df = ta.add_momentum_ta(df, 'High', 'Low', 'Close', 'Volume', fillna=True)
    # df = ta.add_volatility_ta(df, 'High', 'Low', 'Close', fillna=True)
    df = ta.add_trend_ta(df, 'High', 'Low', 'Close', fillna=True)
    df = ta.add_others_ta(df, 'Close', fillna=True)
    return df


def get_banque_data():
    for file_name in os.listdir('../data/banque'):
        yield read_csv(f'../data/banque/{file_name}')


# data = list(get_banque_data())
data = [read_csv('../data/banque/bnp_1h_2d.csv')]

env = TradingEnv(data, initial_account=3000, window_size=20)

nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions, activation='linear'))

print(model.summary())

memory = SequentialMemory(limit=5500, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, target_model_update=1e-2,
               policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=10_000, nb_max_episode_steps=MAX_STEP, visualize=False, verbose=2)
dqn.test(env, nb_episodes=1, visualize=True, nb_max_episode_steps=MAX_STEP)
