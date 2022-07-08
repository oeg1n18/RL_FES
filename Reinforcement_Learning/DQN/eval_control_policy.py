from __future__ import absolute_import, division, print_function
from environment import hand_env

import numpy as np
import matplotlib.pyplot as plt

import os
import tensorflow as tf
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.eval.metric_utils import log_metrics
import logging
from tf_agents.utils.common import function
from tf_agents.utils import common
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.environments import suite_gym


from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
policy_dir = "policy"

from environment import hand_env

avg_returns = np.loadtxt("returns/avg_returns.txt")
plt.plot(avg_returns)
plt.xlabel("Episode (10^1)")
plt.ylabel("Episode Reward")
plt.show()

#env_eval = hand_env()
#tf_env_eval = tf_py_environment.TFPyEnvironment(env_eval)

env_name = 'CartPole-v0'
env_eval = suite_gym.load(env_name)
tf_env_eval = tf_py_environment.TFPyEnvironment(env_eval)

policy = tf.saved_model.load(policy_dir)
print(policy)

obs = []
step = 0
time_step = tf_env_eval.reset()
print(time_step)
while step < 10:
    step += 1
    print("step: ", step)
    time_step = policy.action(time_step)
    time_step = tf_env_eval.step(time_step.action)
    obs.append(time_step)

print("The final input state variables are: ")
print(obs[-1])

