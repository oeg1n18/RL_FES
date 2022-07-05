import itertools
from abc import ABC

import numpy as np
import pandas as pd
import tensorflow as tf

import tf_agents
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tensorflow import keras


class hand_env(tf_agents.environments.py_environment.PyEnvironment, ABC):

    def __init__(self):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=61, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(5,), dtype=np.float32, minimum=[0.0, 0.0, 0.0, 0.0, 0.0], name='observation')
        self.target_trajectory = np.transpose(pd.read_csv("trajectory_target.csv", header=None).to_numpy())
        self.stop_time = 2.5
        self.amp1 = 0.150
        self.amp2 = 0.10
        self.done = 0
        self.reward = 0.0
        self.channel1 = 0.0
        self.channel2 = 1.0
        self._state = np.array([self.stop_time,
                                self.amp1,
                                self.amp2,
                                self.channel1,
                                self.channel2], dtype=np.float32)
        self.hand_model = keras.models.load_model("Models/model_cp.h5")
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.array([2.5, 0.150, 0.10, 0.0, 1.0], dtype=np.float32)
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        stop_time = self._state[0]
        amp1 = self._state[1]
        amp2 = self._state[2]
        chan1 = self._state[3]
        chan2 = self._state[4]

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        # Make sure episodes don't go on forever.
        perms = list(itertools.permutations(np.arange(8.0), 2))
        n_AC = len(perms)

        # take action
        if action < n_AC:
            chan1 = perms[action][0]
            chan2 = perms[action][1]
        elif action == n_AC:
            amp1 += 0.01
        elif action == int(n_AC + 1):
            amp1 -= 0.01
        elif action == int(n_AC + 2):
            amp2 += 0.01
        elif action == int(n_AC + 3):
            amp2 -= 0.01
        elif action == int(n_AC + 4):
            stop_time += 0.1
        elif action == int(n_AC + 5):
            stop_time -= 0.1
        else:
            raise ValueError('`action` should be integer between 0 and 61.')

        inputs = np.zeros((1, 251, 7))
        slope = np.arange(0, 1, 0.02 / self.stop_time)
        initial = np.zeros(np.arange(0, 0.5, 0.02).shape)
        n_slope = initial.size + slope.size
        steady_state = np.ones(251 - n_slope)
        ramp = np.hstack((initial, slope, steady_state))

        if chan1 != 7:
            inputs[0, :, int(chan1)] = ramp * self.amp1
        if chan2 != 7:
            inputs[0, :, int(chan2)] = ramp * self.amp2
        angles = self.hand_model(inputs)
        angles = angles.numpy()
        error = np.sum(np.abs(angles[0, :, :] - self.target_trajectory[:, :]))

        if self._episode_ended or error > 899:
            reward = np.float32((1 / error))
            return ts.termination(self._state, reward)
        else:
            self._state = np.array([stop_time, amp1, amp2, chan1, chan2], dtype=np.float32)
            return ts.transition(self._state, reward=(1 / error), discount=0.99)



