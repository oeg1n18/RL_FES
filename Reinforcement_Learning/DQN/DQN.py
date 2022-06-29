from tensorflow import keras
import numpy as np
from HandModel import HandModel
from collections import deque


class DQN():
    def __init__(self):
        self.batch_size = 32
        self.discount_factor = 0.95
        self.n_actions = 61
        self.n_states = 5
        self.model = keras.models.Sequential([
                    keras.layers.InputLayer(input_shape=[None, self.n_states]),
                    keras.layers.Dense(35),
                    keras.layers.Dense(50),
                    keras.layers.TimeDistributed(keras.layers.Dense(self.n_actions))])
        self.model.compile(loss="mse", optimizer="adam")
        self.experience_replay = {}
        self.n_experiences = 0
        self.max_episodes = 1000
        self.episode = 0
        self.epsilon = 1

    def store(self, state, action, reward, next_state):
        self.n_experiences += 1
        index = str(self.n_experiences)
        self.experience_replay[index] = (state, action, reward, next_state)

    def sample_memory(self):
        indexes = np.random.randint(self.n_experiences, size=self.batch_size)
        samples = []
        for index in indexes:
            samples.append(self.experience_replay[str(index)])
        return samples

    def get_action(self, state):
        if self.episode >= self.max_episodes:
            print("Completed Training")
            return 1
        else:
            self.episode += 1

        values = self.model(state)

        if np.random.rand() < self.epsilon
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(values)
        if self.epsilon > 0:
            self.epsilon -= 1/(0.9*self.max_episodes)
        return action

    def experience_replay(self):
        memories = self.sample_memory()
        Q_preds = np.zeros(self.batch_size)
        Q_targets = np.zeros(self.batch_size)
        states = np.zeros((self.batch_size, self.state))
        for i, memory in enumerate(memories):
            state = memory[0]
            action = memory[1]
            reward = memory[2]
            nstate = memory[3]

            Q_pred = self.model(state)
            Q_preds[i] = Q_pred[action]
            Q_targets[i] = (reward - self.discount_factor * np.max(self.model(nstate)))














