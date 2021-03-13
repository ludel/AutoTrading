import os

import pandas as pd
import ta
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy
# from keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList

from env.trading import TradingEnv

MAX_STEP = 200


def read_csv(file):
    df = pd.read_csv(file).drop('Date', axis=1)
    df = df[df['Open'] != 0]
    return ta.add_all_ta_features(df, 'Open', 'High', 'Low', 'Close', 'Volume', fillna=True)


def get_banque_data():
    for file_name in os.listdir('../data/banque'):
        yield read_csv(f'../data/banque/{file_name}')


# data = list(get_banque_data())
data = [read_csv('../data/banque/bnp_1h_2d.csv')]

env = TradingEnv(data, initial_account=3000, max_step=MAX_STEP, window_size=20)

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=MAX_STEP, visualize=False, verbose=2, nb_max_start_steps=MAX_STEP)
dqn.test(env, nb_episodes=1, visualize=True)
