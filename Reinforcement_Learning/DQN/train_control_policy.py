from environment import hand_env
from DQN import DQN
import matplotlib.pyplot as plt
import tensorflow as tf

RL_agent = DQN()
env = hand_env()
rewards = []
experience_replay_frequency = 10
RL_agent.max_episodes = 100

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


plt.plot(rewards)
plt.show()