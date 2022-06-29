

from HandModel import HandModel
import numpy as np
import pandas as pd
import itertools


class hand_env(HandModel):
    def __init__(self):
        super().__init__()
        self.step = 0
        self.target_trajectory = np.transpose(pd.read_csv("trajectory_target.csv", header=None).to_numpy())
        self.stop_time = 2.5
        self.amp1 = 0.150
        self.amp2 = 0.10
        self.done = 0
        self.reward = 0
        self.channel1 = 0
        self.channel2 = 1
        self.state = np.array([self.stop_time,
                               self.amp1,
                               self.amp2,
                               self.channel1,
                               self.channel2]).astype(np.float32)

    def take_action(self, A):
        old_state = np.reshape(self.state, (1, 5)).astype(np.float32)
        perms = list(itertools.permutations(np.arange(8.0), 2))
        n_AC = len(perms)

        inputs = np.zeros((1, 251, 7))

        # take action
        if A < n_AC:
            self.channel1 = int(perms[A][0])
            self.channel2 = int(perms[A][1])
        if A == n_AC:
            self.amp1 = self.amp1 + 0.01
        if A == n_AC + 1:
            self.amp1 = self.amp1 - 0.01
        if A == n_AC + 2:
            self.amp2 = self.amp2 + 0.01
        if A == n_AC + 3:
            self.amp2 = self.amp2 - 0.01
        if A == n_AC + 4:
            self.stop_time = self.stop_time + 0.1
        if A == n_AC + 5:
            self.stop_time = self.stop_time - 0.1

        # Safety Checks
        if self.stop_time > 3.0:
            self.stop_time = 3.0
        if self.stop_time < 1.0:
            self.stop_time = 1.0
        if self.amp1 > 0.15:
            self.amp1 = 0.15
        if self.amp2 > 0.15:
            self.amp2 = 0.15
        if self.amp1 < 0.025:
            self.amp1 = 0.025
        if self.amp2 < 0.025:
            self.amp2 = 0.025

        slope = np.arange(0, 1, 0.02 / self.stop_time)
        initial = np.zeros(np.arange(0, 0.5, 0.02).shape)
        n_slope = initial.size + slope.size
        steady_state = np.ones(251 - n_slope)
        ramp = np.hstack((initial, slope, steady_state))

        if self.channel1 != 7:
            inputs[0, :, self.channel1] = ramp * self.amp1
        if self.channel2 != 7:
            inputs[0, :, self.channel2] = ramp * self.amp2

        # apply input
        self.reset_model()
        angles = self.infer(inputs)

        # Compute Reward
        self.reward = -np.sum(np.abs(angles[0, 250, :] - self.target_trajectory[250, :]))
        self.reward = np.float32(self.reward)

        # Compute next state
        self.state = np.reshape(np.array([self.stop_time,
                               self.amp1,
                               self.amp2,
                               self.channel1,
                               self.channel2]), (1, 5)).astype(np.float32)
        # Compute Done
        return old_state, A, self.reward, self.state

    def reset(self):
        self.stop_time = 2.5
        self.amp1 = 0.150
        self.amp2 = 0.10
        self.done = 0
        self.reward = 0
        self.channel1 = 0
        self.channel2 = 1
        self.state = np.array([self.stop_time,
                               self.amp1,
                               self.amp2,
                               self.channel1,
                               self.channel2])

