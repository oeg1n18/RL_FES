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


print("Is configuration built for gpu? : ", tf.config.list_physical_devices('GPU'))

NUM_ITERATIONS = 1000  # @param {type:"integer"}
INITIAL_COLLECT_STEPS = 2  # @param {type:"integer"}
REPLAY_BUFFER_LENGTH = 10000  # @param {type:"integer"}
BATCH_SIZE = 3  # @param {type:"integer"}
LEARNING_RATE = 1e-3  # @param {type:"number"}
LOG_INTERVAL = 1  # @param {type:"integer"}
EVAL_INTERVAL = 10  # @param {type:"integer"}

policy_dir = "policy"

# Transform the environment into a tensorflow environment
env_name = 'CartPole-v0'
env = suite_gym.load(env_name)
env_eval = suite_gym.load(env_name)
tf_env_train = tf_py_environment.TFPyEnvironment(env)
tf_env_eval = tf_py_environment.TFPyEnvironment(env_eval)
#env_train = hand_env()
#env_eval = hand_env()
#tf_env_train = tf_py_environment.TFPyEnvironment(env_train)
#tf_env_eval = tf_py_environment.TFPyEnvironment(env_eval)

# Build the Qnetwork for the DQN Agent
q_net = q_network.QNetwork(
    tf_env_train.observation_spec(),
    tf_env_train.action_spec(),
    fc_layer_params=(100,))

q_net_target = q_network.QNetwork(
    tf_env_train.observation_spec(),
    tf_env_train.action_spec(),
    fc_layer_params=(100,))


optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE )
train_step_counter = tf.Variable(0)

# Create the DQN agent
agent = dqn_agent.DqnAgent(
    tf_env_train.time_step_spec(),
    tf_env_train.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    target_q_network=q_net_target,
    train_step_counter=train_step_counter)

# Create replay buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env_train.batch_size,
    max_length=REPLAY_BUFFER_LENGTH)  # reduce if OOM error

# Create the observer
replay_buffer_observer = replay_buffer.add_batch

# Second Observer to show training progress
class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


# Training metrics
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

# Log the train metrics
logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)

# Create the Collect Driver
collect_driver = DynamicStepDriver(
    tf_env_train,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics)

# Collect initial experiences with random policy
initial_collect_policy = RandomTFPolicy(tf_env_train.time_step_spec(),
                                        tf_env_train.action_spec())

# Turn the replay buffer into a dataset for training
dataset = replay_buffer.as_dataset(
    sample_batch_size=BATCH_SIZE,
    num_steps=2,
    num_parallel_calls=3,
    single_deterministic_pass=False).prefetch(3)

# optimize functions by converting them to tensorflow
collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

# Create the step driver by:
# 1. giving it a policy to play with
# 2. giving it a policy to collect the data with
# 3. provide it the observers which show progress and log data in the replay buffer
init_driver = DynamicStepDriver(
    tf_env_train,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(200)],
    num_steps=INITIAL_COLLECT_STEPS )

# Checkpoint saver
tf_policy_saver = policy_saver.PolicySaver(agent.policy)


# Function for computing the average return
def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


# Training Loop
# 1. get the initial policy state
# 2. turn the dataset into an iterator show you can get the "next" batch with "next" method
# 3. use the collect driver to run the policy to get a time_step
def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env_train.batch_size)
    iterator = iter(dataset)
    avg_returns = []
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        reward = time_step.reward.numpy()
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % LOG_INTERVAL == 0:
            log_metrics(train_metrics)
        if iteration % EVAL_INTERVAL == 0:
           avg_returns.append(compute_avg_return(tf_env_eval, agent.policy, 1))
    return avg_returns

print("Hi")
agent.initialize()
print("Hi")
final_time_step, final_policy_state = init_driver.run()  # Collect some initial experiences
print("Hi2")
returns = train_agent(NUM_ITERATIONS)  # Run the training loop
tf_policy_saver.save(policy_dir)
print("Hi3")
np.savetxt("returns/avg_returns.txt", returns)
tf_env_train.close()
tf_env_eval.close()
