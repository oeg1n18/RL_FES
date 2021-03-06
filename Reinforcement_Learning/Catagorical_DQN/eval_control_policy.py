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

from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
policy_dir = "policy"

from environment import hand_env

avg_returns = np.loadtxt("returns/avg_returns.txt")
plt.plot(avg_returns)
plt.show()

env_eval = hand_env()
tf_env_eval = tf_py_environment.TFPyEnvironment(env_eval)

policy = tf.saved_model.load(policy_dir)
print(policy)

obs = []
for _ in range(1):
    time_step = tf_env_eval.reset()
    step = 0
    while not time_step.is_last():
        print("step: ", step)
        step += 100
        action_step = policy.action(time_step)
        time_step = tf_env_eval.step(action_step.action)
        obs.append(time_step)

print("The final input state variables are: ")
print(obs[-1])

