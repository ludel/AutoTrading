import random
from enum import Enum
from pprint import pprint

import gym
import matplotlib.pyplot as plt
import numpy as np


class Actions(Enum):
    Buy = 0
    Sell = 1


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, all_df, initial_account, max_step, window_size, hold_bonus_increment=0.08, hold_bonus_start=0.5):
        print('Setup new env ...')
        self.seed(42)
        self.all_df = all_df
        self.df = random.choice(all_df)

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(window_size, self.df.shape[1]))

        self.initial_account = initial_account
        self.liquidity = initial_account
        self.current_step = 0
        self.total_reward = 0
        self.last_transaction_price = 0
        self.hold_bonus = hold_bonus_start
        self.hold_bonus_increment = hold_bonus_increment
        self.hold_bonus_start = hold_bonus_start

        self.max_step = max_step
        self.end_tick = max_step
        self.window_size = window_size
        self.history = {}

        self.action_owned = 0

    def reset(self):
        self.df = random.choice(self.all_df)
        self.current_step = random.randint(self.window_size, len(self.df) - (self.max_step + 1))
        self.end_tick = self.current_step + self.max_step
        self.total_reward = 0
        self.hold_bonus = self.hold_bonus_start
        self.last_transaction_price = 0
        self.action_owned = 0
        self.liquidity = self.initial_account
        self.history = {}
        print(f'Reset Env - step {self.current_step} to {self.end_tick}')
        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        observation = self._get_observation()
        done = self.current_step == self.end_tick
        step_reward = 0
        current_price = self.df.iloc[self.current_step]['Open']
        info = {'current price': current_price, 'last transaction': self.last_transaction_price,
                'action': 'Ignore'}

        if self.action_owned > 0:
            self.hold_bonus += self.hold_bonus_increment
        else:
            self.hold_bonus = self.hold_bonus_start

        if action == Actions.Buy.value and self.action_owned == 0:
            max_quantity = int(self.liquidity / current_price)
            self.last_transaction_price = current_price
            self.liquidity -= current_price * max_quantity
            self.action_owned = max_quantity
            info['action'] = 'Buy'

        elif action == Actions.Sell.value and self.action_owned > 0:
            self.liquidity += self.action_owned * current_price
            step_reward = (self.action_owned * current_price) - (self.last_transaction_price * self.action_owned)
            self.action_owned = 0
            info['action'] = 'Sell'
            self.last_transaction_price = 0

        step_reward *= self.hold_bonus if step_reward > 0 else 1
        self.total_reward += step_reward
        info['net'] = self.net_worth
        info['portfolio'] = {'action_owned': self.action_owned, 'liquidity': self.liquidity}
        info['reward'] = {'step': step_reward, 'total': self.total_reward, 'hold bonus': self.hold_bonus}
        self.history[self.current_step] = info
        return observation, step_reward, done, info

    @property
    def net_worth(self):
        current_price = self.df.iloc[self.current_step]['Open']
        return self.liquidity + (self.action_owned * current_price)

    def _get_observation(self):
        return self.df[(self.current_step - self.window_size):self.current_step]

    def render(self, mode='human', close=False):
        pprint(self.history[-1])

    def render_all(self):
        pprint(self.history)

    def plot(self):
        plt.figure(figsize=(15, 7))
        sell = {'x': [], 'y': []}
        buy = {'x': [], 'y': []}
        for k, v in self.history.items():
            if v['action'] == 'Sell':
                sell['y'].append(v['current price'])
                sell['x'].append(k)
            if v['action'] == 'Buy':
                buy['y'].append(v['current price'])
                buy['x'].append(k)
        y = [i['current price'] for i in self.history.values()]
        plt.plot(self.history.keys(), y)
        plt.plot(sell['x'], sell['y'], 'ro', markersize=10)
        plt.plot(buy['x'], buy['y'], 'go', markersize=10)

        for i in range(len(sell['x'])):
            plt.annotate(i, xy=(buy['x'][i], buy['y'][i]), weight='heavy', horizontalalignment='center',
                         verticalalignment='center', color='white', fontsize='small')
            plt.annotate(i, xy=(sell['x'][i], sell['y'][i]), weight='heavy', horizontalalignment='center',
                         verticalalignment='center', color='white', fontsize='small')

        plt.show()
