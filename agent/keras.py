import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class Agent:

    def __init__(self, input_dim, output_dim, learning_rate=1e-4):
        self.learning_rate = learning_rate
        self.input_dimension = input_dim
        self.output_dimension = output_dim
        self.model = self._build_policy()

    def _build_policy(self):
        model = Sequential()
        model.add(Dense(20, input_shape=(self.input_dimension,), activation="relu"))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.output_dimension, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='kullback_leibler_divergence', optimizer=opt)
        return model

    def decide(self, state):
        X = state.reshape(1, state.shape[0])
        return self.model.predict(X, batch_size=1)

    def train(self, decision, reward):
        # y_hat = np.around(y_hat)
        decision = np.vstack([decision])
        reward = np.vstack([reward])
        self.model.fit(decision, reward, verbose=0)
