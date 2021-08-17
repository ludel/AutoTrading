import random
import warnings
from enum import Enum
from pprint import pprint

import empyrical
import gym
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from abstract.history import History, Event
from abstract.portfolio import Portfolio


class Actions(Enum):
    Buy = 0
    Sell = 1


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, all_df, initial_account, window_size, reward_method='sharpe_ratio',
                 random_first_step=False):
        print('Setup new env ...')
        self.seed(42)
        self.all_df = all_df
        self.df = random.choice(all_df)

        # spaces
        self.action_space = gym.spaces.Discrete(len(Actions))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(window_size, self.df.shape[1]))

        self.portfolio = Portfolio(initial_account)

        self.random_first_step = random_first_step

        self.reward_method = reward_method
        self.use_empyrical = hasattr(empyrical, reward_method)
        if self.use_empyrical:
            self.reward_method = getattr(empyrical, reward_method)

        self.reward_len = window_size
        self.current_step = window_size
        self.window_size = window_size

        self.history = History()
        self.nb_init_actions = initial_account / self.df.iloc[self.current_step]['Open']

    def empyrical_reward(self):
        net_worths = [h.portfolio['net_worth'] for h in self.history if h.step > self.history.last_buy_step]
        returns = np.diff(net_worths)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward = self.reward_method(returns)
        return reward if reward != np.inf and not np.isnan(reward) else 0

    def sharpe_ration(self, action_name):
        net_worths = [h.portfolio['net_worth'] for h in self.history if h.step > self.history.last_buy_step]
        returns = np.diff(net_worths)
        print(net_worths, action_name, returns, returns.mean() / returns.std() * np.sqrt(252))
        return returns.mean() / returns.std() * np.sqrt(252)

    def calcul_reward(self, action_name):
        if self.history.last_buy_step is None:
            return 0

        if self.use_empyrical:
            reward = self.empyrical_reward()
        else:
            return getattr(self, self.reward_method)(action_name)
        return reward

    def reset(self):
        self.df = random.choice(self.all_df)
        if self.random_first_step:
            self.current_step = random.randint(self.window_size, len(self.df))
        else:
            if self.current_step == len(self.df) - 1:
                self.current_step = self.window_size

        self.portfolio = Portfolio(self.portfolio.initial_account)
        self.history = History()
        print('Reset env start from', self.current_step)
        return self._get_observation()

    def step(self, action):
        self.current_step += 1
        observation = self._get_observation()
        done = self.current_step == len(self.df) - 1

        self.portfolio.current_price = self.df.iloc[self.current_step]['Open']

        step_reward, action_name = self._process_action(action)
        event = Event(step=self.current_step, portfolio=self.portfolio.as_data(), action=action_name,
                      reward=step_reward)
        self.history.add_event(event)
        return observation, step_reward, done, {}

    def _process_action(self, action):

        if action == Actions.Buy.value and self.portfolio.action_owned == 0:
            max_quantity = int(self.portfolio.liquidity / self.portfolio.current_price)
            self.portfolio.liquidity -= self.portfolio.current_price * max_quantity
            self.portfolio.action_owned = max_quantity
            self.history.last_buy_step = self.current_step
            action_name = 'Buy'
        elif action == Actions.Sell.value and self.portfolio.action_owned > 0:
            self.portfolio.liquidity += self.portfolio.action_owned * self.portfolio.current_price
            self.portfolio.action_owned = 0
            action_name = 'Sell'
        elif self.portfolio.action_owned > 0:
            action_name = 'Hold'
        else:
            action_name = 'Ignore'
            self.history.last_buy_step = None

        step_reward = self.calcul_reward(action_name)

        if action_name == 'Sell':
            self.history.last_buy_step = None

        return step_reward, action_name

    def _get_observation(self):
        obs = self.df[(self.current_step - self.window_size):self.current_step]
        return MinMaxScaler().fit_transform(obs)

    def render(self, mode='human', close=False):
        last_step = self.history[-1].reward
        print(f'==> reward step {last_step} - net {self.portfolio.net_worth}')

    def render_all(self):
        pprint(self.history)

    def plot(self, plot_marker=True, plot_marker_index=True):
        plt.figure(figsize=(15, 7))
        X = [h.step for h in self.history]
        first_step = self.history[0].step
        first_price = self.df.iloc[first_step]['Open']
        nb_init_actions = self.portfolio.initial_account / first_price

        plt.plot(X, [nb_init_actions * h.portfolio['current_price'] for h in self.history], 'k-',
                 label='Buy and hold')
        plt.plot(X, [h.portfolio['net_worth'] for h in self.history], 'k--', label='net worth')

        if plot_marker:
            sell = {'x': [], 'y': []}
            buy = {'x': [], 'y': []}
            for history in self.history:
                if history.action == 'Sell':
                    sell['y'].append(history.portfolio['current_price'] * nb_init_actions)
                    sell['x'].append(history.step)
                if history.action == 'Buy':
                    buy['y'].append(history.portfolio['current_price'] * nb_init_actions)
                    buy['x'].append(history.step)
            plt.plot(sell['x'], sell['y'], 'rv', markersize=10)
            plt.plot(buy['x'], buy['y'], 'g^', markersize=10)

            if plot_marker_index:
                for i in range(len(sell['x'])):
                    plt.annotate(i, xy=(buy['x'][i], buy['y'][i] - 17), weight='heavy', horizontalalignment='center',
                                 verticalalignment='center', color='black', fontsize=7)
                    plt.annotate(i, xy=(sell['x'][i], sell['y'][i] - 17), weight='heavy', horizontalalignment='center',
                                 verticalalignment='center', color='black', fontsize=7)

        plt.legend()
        plt.show()
