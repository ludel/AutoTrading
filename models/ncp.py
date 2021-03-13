from collections import deque

import gym
import numpy as np
import torch

from agent.NCP import Agent


class Transition:
    def __init__(self, state, action, new_state, reward, done):
        self.state = state
        self.action = action
        self.new_state = new_state
        self.reward = reward
        self.done = done


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device, '\n')

env = gym.make('CartPole-v0')
agent = Agent(len(env.observation_space.low), env.action_space.n, device)

transitions = deque(maxlen=1_000_000)
num_episodes = 500
chkpt_every = 20
sum_rewards = []

batch_size = 8
min_transitions = 1_000

for episode in range(1, num_episodes):

    state = env.reset()
    done = False
    ep_rewards = []

    while not done:
        action = agent.get_action(state)

        new_state, reward, done, _ = env.step(action)
        ep_rewards.append(reward)

        transitions.append(Transition(state, action, new_state, reward, done))
        state = new_state

        if len(transitions) > min_transitions:
            agent.train(np.random.choice(transitions, batch_size, replace=False))

    sum_rewards.append(np.sum(ep_rewards))

    if episode % chkpt_every == 0:
        avg_reward = np.average(sum_rewards)
        print(f'step {episode} avg reward: {avg_reward}')
        sum_rewards = []


