# source: https://arxiv.org/pdf/1811.07522.pdf & https://github.com/AI4Finance-LLC/FinRL

# multiple stock datasets linked by date
# multiple actions http://finrl.org/examples/multiplestocktrading.html
# using DDPG (ElegantRL)
import warnings
from enum import Enum

import empyrical
import gym
import matplotlib.pyplot as plt
import numpy as np

from abstract.history import HistoryImprove, EventImproved
from abstract.portfolio import ImprovedPortfolio
from config import cac_tickers

STOCK_DIM = len(cac_tickers)


class Actions(Enum):
    Buy = 0
    Hold = 1
    Sell = 2


class TradingEnvV2(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_account):
        print('Setup new env ...')
        self.seed(42)
        self.df = df
        self.cash_penalty_proportion = 0.1
        self.window_day = 30
        self.companies_code = cac_tickers

        # spaces
        self.action_space = gym.spaces.MultiDiscrete([len(Actions)] * STOCK_DIM)
        # self.observation_space = gym.spaces.Box(low=0, high=1, shape=(window_size, self.df.shape[1]))
        nb_feature = df.shape[1] - 1  # remove Code column
        window_size = STOCK_DIM * (self.window_day + 1)

        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(window_size, nb_feature))

        self.portfolio = ImprovedPortfolio(initial_account)
        self.portfolio.init_action_portfolio(cac_tickers)

        self.terminal = False
        self.history = HistoryImprove()

        self.initial_day = df.index[0] + self.window_day
        self.current_day = self.initial_day

    def step(self, actions):
        self.current_day += 1
        if self.current_day % 100 == 0:
            print(self.current_day)

        current_data = self._get_current_data()
        done = self.current_day == self.df.index[-1]
        if done:
            print(f"Done day {self.current_day}")
        empty_data_index = []

        for index, current_company in enumerate(cac_tickers):
            close_price = current_data.loc[current_data['Code'] == current_company]['Close']
            if close_price.empty:
                empty_data_index.append(index)
                continue
            self.portfolio.set_action_price(current_company, float(close_price))
        sell_index = [i for i, v in enumerate(actions) if v == Actions.Sell.value and i not in empty_data_index]
        buy_index = [i for i, v in enumerate(actions) if v == Actions.Buy.value and i not in empty_data_index]

        for index in sell_index:
            self._sell_stock(cac_tickers[index])

        investment_proportion_amount = self.portfolio.liquidity / len(buy_index)
        for index in buy_index:
            self._buy_stock(investment_proportion_amount, cac_tickers[index])

        step_reward = self._get_reward()
        event = EventImproved(
            day=self.current_day, sell_index=sell_index, portfolio_value=self.portfolio.net_worth,
            buy_index=buy_index, portfolio_action=self.portfolio.action_portfolio.export_prices(), reward=step_reward
        )
        self.history.add(event)
        return self._get_observation(), step_reward, done, {}

    def _buy_stock(self, investment_amount, current_company):
        current_action_price = self.portfolio.get_action_price(current_company)
        buyable_quantity_action = int(investment_amount / current_action_price)
        self.portfolio.buy(current_company, buyable_quantity_action)

    def _sell_stock(self, current_company):
        self.portfolio.sell(current_company)

    def _get_reward(self):
        end_window = self.current_day - self.initial_day
        start_window = end_window - self.window_day
        windows_history = self.history[start_window:end_window]
        net_worths = [e.portfolio_value for e in windows_history]
        returns = np.diff(net_worths)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reward = empyrical.sortino_ratio(returns)

        return reward if reward != np.inf and not np.isnan(reward) else 0

    def _get_observation(self):
        obs = self.df.loc[(self.current_day - self.window_day):self.current_day]
        return obs.drop('Code', axis=1)

    def _get_current_data(self):
        return self.df.loc[self.current_day]

    def reset(self):
        self.initial_day = self.df.index[0] + self.window_day
        self.current_day = self.initial_day
        self.history = HistoryImprove()
        self.portfolio = ImprovedPortfolio(self.portfolio.initial_account)
        self.portfolio.init_action_portfolio(self.companies_code)
        print('Reset env start from ', self.current_day)
        return self._get_observation()

    def plot(self, limit_stock=5):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 7), dpi=300, gridspec_kw={'height_ratios': [3, 1, 1]})
        X = [h.day for h in self.history]
        first_prices = self.history.get_first_prices()
        proportion_amount = self.portfolio.initial_account / len(first_prices)
        nb_init_actions = proportion_amount / np.array(first_prices)

        ax1.plot(X, self.history.get_portfolio_action_owned(nb_init_actions), 'k-', label='Buy and hold')

        prices_by_code = {}
        for code in self.history.get_all_code()[:limit_stock]:
            prices_by_code[code] = self.history.get_all_price_by_code(code, self.portfolio.initial_account)
            ax1.plot(X, prices_by_code[code], label=code, alpha=0.4)

        sell = {'x': [], 'y': []}
        buy = {'x': [], 'y': []}
        for index, event in enumerate(self.history):

            for sell_index in event.sell_index:
                code = cac_tickers[sell_index]
                if not prices_by_code.get(code):
                    continue
                sell['y'].append(prices_by_code[code][index])
                sell['x'].append(event.day)

            for buy_index in event.buy_index:
                code = cac_tickers[buy_index]
                if not prices_by_code.get(code):
                    continue
                buy['y'].append(prices_by_code[code][index])
                buy['x'].append(event.day)

        ax1.plot(sell['x'], sell['y'], 'rv', markersize=1, alpha=0.7)
        ax1.plot(buy['x'], buy['y'], 'g^', markersize=1, alpha=0.7)

        ax2.plot(X, self.history.get_portfolio_values(), 'k--', label='net worth')

        ax3.plot(X, self.history.get_all_rewards(), 'k-', label='reward')
        ax3.legend()
        ax1.legend()
        ax2.legend()
        plt.show()
