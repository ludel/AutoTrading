import itertools

from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import MaxBoltzmannQPolicy

from env.trading import TradingEnv
from preprocessing.feature import read_csv

MAX_STEP = 300
MAX_EPISODE = 2

train_data = [read_csv('../data/banque/bnp_1h_2d.csv'), read_csv('../data/banque/gle_1h_2d.csv'),
              read_csv('../data/banque/aca_1h_2d.csv')]
test_data = read_csv('../data/banque/kn_1h_2d.csv')

metric_choices = ['mse']
eps_choices = [.1]
windows_choices = [20]
reward_method_choices = ['sortino_ratio']
lr_choices = [0.001]


def create_train(metric, eps, window_size, reward_method, lr):
    env = TradingEnv(train_data, initial_account=20_000, window_size=window_size, random_first_step=False,
                     reward_method=reward_method)
    nb_actions = env.action_space.n

    model = Sequential()
    model.add(Flatten(input_shape=(window_size,) + env.observation_space.shape))
    model.add(Dense(24))
    model.add(Activation('relu'))
    model.add(Dense(48))
    model.add(Activation('relu'))
    model.add(Dense(96))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions, activation='linear'))

    memory = SequentialMemory(limit=5000, window_length=window_size)
    policy = MaxBoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, target_model_update=1e-2,
                   policy=policy)
    dqn.compile(Adam(lr), metrics=[metric])

    dqn.fit(env, nb_steps=50_000, visualize=False, verbose=2)
    env.current_step = env.window_size
    env.random_first_step = False
    env.all_df = [test_data]
    env.reset()
    dqn.test(env, nb_episodes=1, visualize=False)

    return env.portfolio.net_worth, dqn, env


if __name__ == '__main__':
    best_score = 0
    best_agent = None
    best_env = None
    product = [metric_choices, eps_choices, windows_choices, reward_method_choices, lr_choices]
    for combination in itertools.product(*product):
        net_worth, current_agent, current_env = create_train(*combination)
        print('Model new worth ', net_worth)
        if net_worth > best_score:
            print('===> New best score {}'.format(net_worth), *combination, sep=' - ')
            best_score = net_worth
            best_agent = current_agent
            best_env = current_env
            current_env.plot(plot_marker=False)
