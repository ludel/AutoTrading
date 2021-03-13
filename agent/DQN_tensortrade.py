import itertools

import numpy as np
import tensorflow as tf
from tensortrade.agents import DQNAgent, DQNTransition
from tensortrade.agents import ReplayMemory


class DQN(DQNAgent):
    def train(self, n_steps: int = None, n_episodes: int = None, save_every: int = None, save_path: str = None,
              callback: callable = None, cyclic_steps=True, **kwargs) -> float:
        batch_size: int = kwargs.get('batch_size', 128)
        discount_factor: float = kwargs.get('discount_factor', 0.9999)
        learning_rate: float = kwargs.get('learning_rate', 0.0001)
        eps_start: float = kwargs.get('eps_start', 0.9)
        eps_end: float = kwargs.get('eps_end', 0.05)
        eps_decay_steps: int = kwargs.get('eps_decay_steps', 200)
        update_target_every: int = kwargs.get('update_target_every', 1000)
        memory_capacity: int = kwargs.get('memory_capacity', 1000)
        render_interval: int = kwargs.get('render_interval', 50)  # in steps, None for episode end renderers only

        memory = ReplayMemory(memory_capacity, transition_type=DQNTransition)
        episode = 0
        total_steps_done = 0
        total_reward = 0
        stop_training = False

        start_step = itertools.cycle(range(0, kwargs.get('max_exploration', 4000), n_steps))
        if n_steps and not n_episodes:
            n_episodes = np.iinfo(np.int32).max

        print('==== AGENT ID: {} ===='.format(self.id))
        while episode < n_episodes and not stop_training:
            if cyclic_steps:
                self.env.clock.start = next(start_step)
            state = self.env.reset()
            done = False
            steps_done = 0
            print('== EPISODE ID ({}/{}): {}: cycle {} =='.format(
                episode + 1, n_episodes, self.env.episode_id, self.env.clock.start
            ))

            while not done:
                threshold = eps_end + (eps_start - eps_end) * np.exp(-total_steps_done / eps_decay_steps)
                action = self.get_action(state, threshold=threshold)
                next_state, reward, done, _ = self.env.step(action)

                memory.push(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps_done += 1
                total_steps_done += 1

                if len(memory) < batch_size:
                    continue

                self._apply_gradient_descent(memory, batch_size, learning_rate, discount_factor)

                if n_steps and steps_done >= n_steps:
                    done = True

                if render_interval is not None and steps_done % render_interval == 0:
                    self.env.render(
                        episode=episode,
                        max_episodes=n_episodes,
                        max_steps=n_steps
                    )

                if steps_done % update_target_every == 0:
                    self.target_network = tf.keras.models.clone_model(self.policy_network)
                    self.target_network.trainable = False

            is_checkpoint = save_every and episode % save_every == 0

            if save_path and (is_checkpoint or episode == n_episodes - 1):
                self.save(save_path, episode=episode)

            episode += 1

        mean_reward = total_reward / steps_done

        return mean_reward
