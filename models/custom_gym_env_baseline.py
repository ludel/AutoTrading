import numpy as np
import optuna

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from env.trading import TradingEnv
from preprocessing.feature import read_csv

data = [read_csv('../data/banque/bnp_1h_2d.csv')]

reward_function = ['sharpe_ratio', 'sortino_ratio', 'omega_ratio']


def optimize_envs(trial):
    return {
        'window_size': int(trial.suggest_loguniform('window_size', 1, 200)),
        'reward_len': int(trial.suggest_loguniform('reward_len', 1, 200)),
        'reward_method': trial.suggest_categorical('reward_method', reward_function)
    }


def optimize_agent(trial):
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 20, len(data[0]) - 20)),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
    }


# data = list(get_banque_data())


def objective_fn(trial):
    env_params = optimize_envs(trial)
    agent_params = optimize_agent(trial)
    print(env_params)
    print(agent_params)

    env = TradingEnv(data, initial_account=3000, window_size=env_params['window_size'],
                     reward_len=env_params['reward_len'], reward_method=env_params['reward_method'])

    model = PPO(MlpPolicy, env, **agent_params)

    model.learn(10000)

    rewards, done = [], False

    obs = env.reset()
    env.current_step = 20
    for i in range(env.current_step, len(env.df)-100):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
    print(rewards)
    return -np.mean(rewards)


def optimize(n_trials=500, n_jobs=4):
    study = optuna.create_study(study_name='optimize_profit', storage='sqlite:///params.db', load_if_exists=True)
    study.optimize(objective_fn, n_trials=n_trials, n_jobs=n_jobs)


optimize(n_jobs=4)
