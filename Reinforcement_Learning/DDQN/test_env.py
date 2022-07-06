from __future__ import absolute_import, division, print_function
from environment import hand_env

import numpy as np
import logging

import tensorflow as tf
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.utils.common import function
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import suite_gym
from tf_agents.environments import utils

env_train = hand_env()
env_eval = hand_env()

utils.validate_py_environment(env_train, episodes=5)

env_train.close()
env_eval.close()