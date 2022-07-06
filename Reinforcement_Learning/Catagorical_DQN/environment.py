import itertools
from abc import ABC

import numpy as np
import pandas as pd
import tensorflow as tf

import tf_agents
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tensorflow import keras
from tf_agents.environments import utils
from tf_agents.environments import suite_gym
import time

TIME = False
TEST = False


class hand_env(tf_agents.environments.py_environment.PyEnvironment, ABC):

    def __init__(self):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=9, name='action')
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
        self._state = np.array([2.5, 0.150, 0.10, 0.0, 1.0], dtype=np.float32)
        self.hand_model = keras.models.load_model("Models/model_cp.h5")
        self._episode_ended = False
        self.counter = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.array([2.5, 0.150, 0.10, 0.0, 1.0], dtype=np.float32)
        self._episode_ended = False
        self.counter = 0
        return ts.restart(self._state)

    def _step(self, action):
        self.counter += 1
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
        if action == 0:
            chan1 += 1.0
        elif action == 1:
            chan1 -= 1.0
        elif action == 2:
            chan2 += 1.0
        elif action == 3:
            chan2 -= 1.0
        elif action == 4:
            amp1 += 0.01
        elif action == 5:
            amp1 -= 0.01
        elif action == 6:
            amp2 += 0.01
        elif action == 7:
            amp2 -= 0.01
        elif action == 8:
            stop_time += 0.1
        elif action == 9:
            stop_time -= 0.1
        else:
            raise ValueError('`action` should be integer between 0 and 9')

        inputs = np.zeros((1, 251, 7))
        slope = np.arange(0, 1, 0.02 / self.stop_time)
        initial = np.zeros(np.arange(0, 0.5, 0.02).shape)
        n_slope = initial.size + slope.size
        steady_state = np.ones(251 - n_slope)
        ramp = np.hstack((initial, slope, steady_state))

        if chan1 < 7 and chan1 >= 0:
            inputs[0, :, int(chan1)] = ramp * self.amp1
        if chan2 < 7 and chan2 >= 0:
            inputs[0, :, int(chan2)] = ramp * self.amp2
        angles = self.hand_model(inputs)
        angles = angles.numpy()
        error = np.sum(np.abs(angles[0, :, :] - self.target_trajectory[:, :]))
       # print("error ", error, "  counter", self.counter, " ended ", self._episode_ended)
        if self._episode_ended or error > 890 or stop_time < 0.04 or chan1 < 0 or chan2 < 0 or chan1 > 7 or chan2 > 7 or amp1 > 0.15 or amp2 > 0.15 or amp1 < 0 or amp2 < 0 or self.counter > 20:
            reward = np.float32((1 / error))
            return ts.termination(self._state, reward)
        else:
            self._state = np.array([stop_time, amp1, amp2, chan1, chan2], dtype=np.float32)
            return ts.transition(self._state, reward=(1 / error), discount=0.99)


henv = hand_env()
env = suite_gym.load('CartPole-v0')

if TIME:
    t0 = time.time()
    utils.validate_py_environment(henv, episodes=5)
    t1 = time.time()
    print("Hand Environment took: ", t1 - t0)
    t2 = time.time()
    utils.validate_py_environment(env, episodes=5)
    t3 = time.time()
    print("openai env took: ", t3 - t2)


if TEST:
    environment = hand_env()
    tot_rewards = []
    for episode in range(20):
        time_step = environment.reset()
        cumulative_reward = time_step.reward
        step = 0
        while time_step.is_last() == False:
            print("episode: ", episode, " step: ", step)
            action = np.random.randint(0, 9)
            action = np.array(action , dtype=np.int32)
            time_step = environment.step(action)
            cumulative_reward += time_step.reward
            step += 1
        tot_rewards.append(cumulative_reward)
    print(tot_rewards)



