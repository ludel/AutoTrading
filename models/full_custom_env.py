# from math import floor
# from random import randint
import numpy as np

from abstract.market import Market
from abstract.portfolio import Portfolio
from agent.keras import Agent
from env.custom import Environment

# Model hyperparameters

num_iterations = 1
market = Market('../data/IntelDataSet.csv')
# signals = market.signals.shape[1]
# market_size = market.asset_prices.shape[0]
agent = Agent(
    input_dim=market.number_of_signals + market.number_of_assets,
    output_dim=market.number_of_assets,
    learning_rate=1e-1
)
running_reward = []

# Output for further analysis, should also output model parameters
# output = open("{}".format(datetime.now().strftime('%Y/%m/%d %H:%M:%S')), "w")

for j in range(num_iterations):

    session = Environment(
        portfolio=Portfolio(cash=5000, market_size=market.number_of_assets),
        agent=agent,
        market=market,
        start_day=2,
        duration=20,
    )
    agent, asset = session.run()

    running_reward.append(asset)

    if j % 10 == 0:
        mean = np.mean(running_reward)
        std = np.std(running_reward)
        running_reward = []
        print("Average cumulated asset value (round {}):\nmean: {:10.2f}\nstd: {:10.2f}".format(j, mean, std))
        # output.write("{:10.2f}\n".format(mean))
