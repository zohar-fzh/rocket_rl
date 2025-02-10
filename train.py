
import matplotlib.pyplot as plt
import torch
import gym
import numpy as np
from PPOAgent import PPOAgent

class Memory:
    def __init__(self):
        self.actions = []   # 行动(共4种)
        self.states = []    # 状态, 由8个数字组成
        self.logprobs = []  # 概率
        self.rewards = []   # 奖励
        self.is_dones = []  ## 游戏是否结束 is_terminals?

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_dones[:]

load_model = True
agent = PPOAgent(load_model)
memory = Memory()
rewards_list = []

env = gym.make('LunarLander-v2', render_mode='rgb_array')

for i in range(3000):
    rewards = []
    state = env.reset()[0]
    while True:
        action, action_prob = agent.act(state)              ### 按照策略网络输出的概率随机采样一个动作
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_prob)
        next_state, reward, done, _, _ = env.step(action)   ### 与环境state进行交互，输出reward 和 环境next_state
        state = next_state
        rewards.append(reward)                              ### 记录每一个动作的reward
        memory.rewards.append(reward)
        memory.is_dones.append(done)

        if len(memory.rewards) >= 1200:
            agent.update(memory)
            memory.clear_memory()

        if done or len(rewards) > 1024:
            rewards_list.append(np.sum(rewards))
            break

    if i%5 == 0:
        print(f"epoch: {i} ,rewards looks like ", rewards_list[-1])

plt.plot(range(len(rewards_list)), rewards_list)
plt.show()
plt.close()

agent.save()

