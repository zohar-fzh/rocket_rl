
from algorithms.PPOAgent import PPOAgent
import gym
import time
import numpy as np
from constants import kUpdateEpochNum, kTimeoutEpochNum

model_path = "export/model_257.pth"
agent = PPOAgent(model_path)
agent.actor_critic.eval()

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

        if step >= kTimeoutEpochNum:
            print("timeout!")
            break

        if terminated:
            break

        time.sleep(0.01)
    print("total reward: ", np.sum(rewards))
    print('Game Ended')
