
import torch
from torch.distributions import Categorical
import torch.optim.lr_scheduler as lr_scheduler
from modules.ActorCritic import ActorCritic

class PPOAgent:
    def __init__(self, load_model=False, lr=0.002, gamma=0.99, K_epochs=5, eps_clip=0.2):
        self.lr = lr  # 学习率
        self.gamma = gamma  # gamma
        self.eps_clip = eps_clip  # 裁剪, 限制值范围
        self.K_epochs = K_epochs  # 获取的每批次的数据作为训练使用的次数

        self.actor_critic = ActorCritic(load_model)

        self.optimizer = torch.optim.Adam(
            [
                {"params":self.actor_critic.actor_layers.parameters()},
                {"params":self.actor_critic.critic_layers.parameters()}
            ],
            lr=lr
        )
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.99)

        self.MseLoss = torch.nn.MSELoss()

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            action_probs = self.actor_critic.actor_layers(state)
            dist = Categorical(action_probs)
            action = dist.sample()                  #根据action_probs的分布抽样
            return action.item(), dist.log_prob(action)

    def evaluate(self, state, action):
        action_probs = self.actor_critic.actor_layers(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.actor_critic.critic_layers(state)
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

            print('Current learning rate:', self.optimizer.param_groups[0]['lr'])

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.scheduler.step()

    def save(self):
        torch.save(self.actor_critic.actor_layers.state_dict(), 'actor_layers.pth')
        torch.save(self.actor_critic.critic_layers.state_dict(), 'critic_layers.pth')
