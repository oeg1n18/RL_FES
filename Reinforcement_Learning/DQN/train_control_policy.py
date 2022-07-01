from environment import hand_env
from DQN import DQN
import tensorflow as tf
from tensorflow import keras
import csv

RL_agent = DQN()
env = hand_env()
rewards = []
experience_replay_frequency = 10
RL_agent.max_episodes = 20

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

while RL_agent.episode < RL_agent.max_episodes:

    state = env.state
    action = RL_agent.get_action(state)
    state, action, reward, nstate = env.take_action(action)
    rewards.append(reward)
    RL_agent.store(state, action, reward, nstate)
    print("episode: ", RL_agent.episode, "  epsilon: ", RL_agent.epsilon, "reward: ", reward)
    if RL_agent.episode%experience_replay_frequency == 0:
        print("Optimizing")
        RL_agent.experience_replay()

print("Finished Training")

with open('rewards.csv', 'w', newline='') as rewards_file:
    input_writer = csv.writer(rewards_file)
    input_writer.writerow(rewards)

policy_model = RL_agent.model
keras.models.save_model(policy_model, "Policy_Models/policy_model.h5")
print("model save complete")
