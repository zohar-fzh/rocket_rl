
import torch
from torch.distributions import Categorical

class Action(torch.nn.Module):
    def __init__(self, state_dim=8, action_dim=4):
        super().__init__()
        self.action_layer = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, state):
        action_logits = self.action_layer(state)
        return action_logits

class Value(torch.nn.Module):
    def __init__(self, state_dim=8):
        super().__init__()
        self.value_layer = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, state):
        state_value = self.value_layer(state)
        return state_value

class PPOAgent:
    def __init__(self, load_model=False, lr=0.002, gamma=0.99, K_epochs=5, eps_clip=0.2):
        self.lr = lr  # 学习率
        self.gamma = gamma  # gamma
        self.eps_clip = eps_clip  # 裁剪, 限制值范围
        self.K_epochs = K_epochs  # 获取的每批次的数据作为训练使用的次数

        self.action_layer = Action()
        self.value_layer = Value()

        if load_model:
            self.action_layer.load_state_dict(torch.load('action_layer.pth'))
            self.value_layer.load_state_dict(torch.load('value_layer.pth'))
            print("model loaded")

        self.optimizer = torch.optim.Adam(
            [
                {"params":self.action_layer.parameters()},
                {"params":self.value_layer.parameters()}
            ],
            lr=lr
        )

        self.MseLoss = torch.nn.MSELoss()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state)
        return action_logprobs, dist_entropy, torch.squeeze(state_value)

    def update(self, memory):
        rewards = []
        temp_reward = 0       
        for reward, is_done in zip(reversed(memory.rewards), reversed(memory.is_dones)):
            if is_done:
                temp_reward = 0
            # 奖励 = 当前状态奖励+0.99*下一状态奖励
            temp_reward = reward + (self.gamma * temp_reward)
            rewards.insert(0, temp_reward)
        # 标准化
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.tensor(memory.states)
        old_actions = torch.tensor(memory.actions)
        old_logprobs = torch.tensor(memory.logprobs)

        for _ in range(self.K_epochs):
            logprobs, dist_entropy, state_values  = self.evaluate(old_states, old_actions)

            ratios =  torch.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def act(self, state):
        state = torch.from_numpy(state).float()
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()                  #根据action_probs的分布抽样
        return action.item(), dist.log_prob(action)

    def save(self):
        torch.save(self.action_layer.state_dict(), 'action_layer.pth')
        torch.save(self.value_layer.state_dict(), 'value_layer.pth')
