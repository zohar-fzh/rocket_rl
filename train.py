
import matplotlib.pyplot as plt
import gym
import numpy as np
from algorithms.PPOAgent import PPOAgent
from memory.Memory import Memory
from constants import kUpdateEpochNum, kTimeoutEpochNum

kEpochNum = 3000

# model_path = "export/model_227.pth"
# agent = PPOAgent(model_path)
agent = PPOAgent()
best_reward = agent.best_reward

memory = Memory()
rewards_list = []

env = gym.make('LunarLander-v2', render_mode='rgb_array')

for i in range(kEpochNum):
    state = env.reset()[0]
    rewards = []    
    while True:
        action, action_prob = agent.act(state)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_prob)

        next_state, reward, done, _, _ = env.step(action)

        memory.rewards.append(reward)
        memory.is_dones.append(done)

        state = next_state
        rewards.append(reward)

        if len(memory.rewards) >= kUpdateEpochNum:
            agent.update(memory)
            memory.clear_memory()

        if len(rewards) > kTimeoutEpochNum:
            rewards_list.append(-1)
            break

        if done:
            rewards_list.append(np.sum(rewards))
            break

    rewards_now = rewards_list[-1]
    if rewards_now > best_reward:
        print("saving best rewards ", rewards_now, " in epoch ", i)
        best_reward = rewards_now
        agent.save(rewards_now)

    if i%10==0:
        print("epoch ", i, " rewards like: ", rewards_now)

plt.plot(range(len(rewards_list)), rewards_list)
plt.savefig(
    'results.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.show()
plt.close()
