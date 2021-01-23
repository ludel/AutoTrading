from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensortrade.agents import A2CAgent
from tensortrade.agents import ReplayMemory

A2CTransition = namedtuple('A2CTransition', ['state', 'action', 'reward', 'done', 'value'])


class A2C(A2CAgent):
    def train(self,
              n_steps: int = None,
              n_episodes: int = None,
              save_every: int = None,
              save_path: str = None,
              callback: callable = None,
              **kwargs) -> float:

        batch_size: int = kwargs.get('batch_size', 128)
        discount_factor: float = kwargs.get('discount_factor', 0.9999)
        learning_rate: float = kwargs.get('learning_rate', 0.0001)
        eps_start: float = kwargs.get('eps_start', 0.9)
        eps_end: float = kwargs.get('eps_end', 0.05)
        eps_decay_steps: int = kwargs.get('eps_decay_steps', 200)
        entropy_c: int = kwargs.get('entropy_c', 0.0001)
        memory_capacity: int = kwargs.get('memory_capacity', 1000)
        render_interval: int = kwargs.get('render_interval', 50)  # in steps, None for episode end render only

        memory = ReplayMemory(memory_capacity, transition_type=A2CTransition)
        episode = 0
        total_steps_done = 0
        total_reward = 0

        if n_steps and not n_episodes:
            n_episodes = np.iinfo(np.int32).max

        print('====      AGENT ID: {}      ===='.format(self.id))

        while episode < n_episodes:
            state = self.env.reset()
            done = False
            steps_done = 0

            print('====      EPISODE ID ({}/{}): {}      ===='.format(episode + 1,
                                                                      n_episodes,
                                                                      self.env.episode_id))

            while not done:
                threshold = eps_end + (eps_start - eps_end) * np.exp(-total_steps_done / eps_decay_steps)
                action = self.get_action(state, threshold=threshold)
                next_state, reward, done, _ = self.env.step(action)

                value = self.critic_network(state[None, :], training=False)
                value = tf.squeeze(value, axis=-1)

                memory.push(state, action, reward, done, value)

                state = next_state
                total_reward += reward
                steps_done += 1
                total_steps_done += 1

                if len(memory) < batch_size:
                    continue

                self._apply_gradient_descent(memory,
                                             batch_size,
                                             learning_rate,
                                             discount_factor,
                                             entropy_c)

                if n_steps and steps_done >= n_steps:
                    done = True

            is_checkpoint = save_every and episode % save_every == 0

            if not render_interval or steps_done < n_steps:
                self.env.render(episode)  # render final state at episode end if not rendered earlier

            self.env.save()

            if save_path and (is_checkpoint or episode == n_episodes - 1):
                self.save(save_path, episode=episode)

            episode += 1

        mean_reward = total_reward / steps_done

        return mean_reward
