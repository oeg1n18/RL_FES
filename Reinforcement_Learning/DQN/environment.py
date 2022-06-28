import itertools

from tensorflow import keras
from HandModel import HandModel
import numpy as np
import pandas as pd
import itertools

class hand_env(HandModel):
    def __init__(self):
        super().__init__()
        self.step = 0
        self.max_steps = 250
        self.state = np.array([2.0944, 0, 1.5708])
        self.input = np.zeros((1, 251, 7))
        self.delta = 0.0015
        self.target_trajectory = np.transpose(pd.read_csv("trajectory_target.csv", header=None).to_numpy())
        self.start_time = 0.5
        self.stop_time = 1.0
        self.amp1 = 0.150
        self.amp2 = 0.150
        self.done = 0
        self.reward = 0

    def take_action(self, A):
        old_state = self.state
        perms = itertools.permutations(np.arange(8.0, 2))
        n_AC = len(perms)

        if A < n_AC:
            channel1 = perms[A,0]
            channel2 = perms[A,1]






        if self.step > self.max_steps:
            self.done = 1
        elif np.max(np.abs(self.state)) > 1.0:
            self.done = 1

        self.reward = -np.sum(np.abs(self.state - self.target_trajectory[self.step, :]))
        return old_state, A, self.reward, self.state, self.done

    def reset(self):
        self.step = 0
        self.reset_model()
        self.state = np.array([2.0944, 0, 1.5708])
        self.input = np.zeros((1, 1, 7))


env = hand_env()

state, action, reward, next_state, done = env.take_action(1)
print(state)
print(action)
print(reward)
print(next_state)
print(done)
