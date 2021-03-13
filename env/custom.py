import numpy as np


class Environment:

    def __init__(self, portfolio, agent, market, start_day=0, trade_cost=0.002, sanction=0.05, duration=500):
        self.portfolio = portfolio
        self.agent = agent
        self.market = market
        self.start_day = start_day
        self.end_day = self.start_day + duration
        self.trade_cost = trade_cost
        self.sanction = sanction

    def run(self):
        for i in range(0, self.end_day):
            state = np.concatenate((self.market.signals[i][:], self.portfolio.weights[i][:]))
            decision = self.agent.decide(state)
            print("decision: ", decision)
            self.portfolio.update_transaction(decision, self.market.asset_prices[i][:])
            asset_returns = self.market.asset_prices[i + 1] / self.market.asset_prices[i] - 1
            self.portfolio.set_portfolio_return(asset_returns)

            reward = self.compute_reward(i, asset_returns)
            print("reward: ", reward)
            self.agent.train(state, reward)

        print("last decision", decision)
        print("last rew", reward)
        print("last quantities: ", self.portfolio.quantities[-1])

        asset = self.portfolio.get_total_value(self.market.asset_prices[i][:])
        return self.agent, asset

    def compute_reward(self, i, asset_returns, reward_learning_rate=0.5):
        sharpe_derivative = self.portfolio.process_sharpe_ratio()
        reward = (self.portfolio.weights[-1] + reward_learning_rate * sharpe_derivative * asset_returns)
        # print("before mod reward:", reward)
        # the following softmax operation makes sure the weights are positive and sum to 1
        return np.exp(reward) / np.sum(np.exp(reward))
