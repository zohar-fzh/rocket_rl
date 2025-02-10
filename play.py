
from PPOAgent import PPOAgent
import gym
import time
import numpy as np

load_model = True
agent = PPOAgent(load_model)

agent.action_layer.eval()
agent.value_layer.eval()

env = gym.make('LunarLander-v2', render_mode='human')

for episode in range(5):
    state = env.reset()[0]
    step = 0
    rewards = []
    while True:
        action, action_prob = agent.act(state)
        state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        step += 1
        if terminated or step >= 600:
            break
        time.sleep(0.01)
    print("total reward: ", np.sum(rewards))
    print('Game Ended')
