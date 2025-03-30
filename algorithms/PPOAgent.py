
import torch
from torch.distributions import Categorical
import torch.optim.lr_scheduler as lr_scheduler
from modules.ActorCritic import ActorCritic
import numpy as np

class PPOAgent:
    def __init__(self, model_path="", lr=0.002, gamma=0.99, update_epochs=3, eps_clip=0.2):
        self.actor_critic = ActorCritic()
        self.best_reward = 0
        if model_path != "":
            checkpoint = torch.load(model_path)
            self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
            self.best_reward = checkpoint["reward"]
            print("loaded model from: ", model_path, " ,reward: ", self.best_reward)

        self.MseLoss = torch.nn.MSELoss()

        self.writer = None

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.update_epochs = update_epochs
        self.gae_lambda = 0.92
        self.optimizer = torch.optim.Adam(
            [
                {"params":self.actor_critic.actor_layers.parameters()},
                {"params":self.actor_critic.critic_layers.parameters()}
            ],
            lr=lr
        )

        self.lr = lr        
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.99)

        return

    def set_writer(self, writer):
        self.writer = writer
        return

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            action_probs = self.actor_critic.actor_layers(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action)

    def evaluate(self, state, action):
        action_probs = self.actor_critic.actor_layers(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.actor_critic.critic_layers(state)
        return action_logprobs, dist_entropy, torch.squeeze(state_value)

    def update(self, memory, iteration):
        old_states = torch.tensor(np.array(memory.states), dtype=torch.float32)
        old_actions = torch.tensor(memory.actions, dtype=torch.long)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)
        rewards = torch.tensor(memory.rewards, dtype=torch.float32)
        is_dones = torch.tensor(memory.is_dones, dtype=torch.float32)

        with torch.no_grad():
            state_values = self.actor_critic.critic_layers(old_states).squeeze()

        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0.0
        next_value = 0.0
        next_non_terminal = 1.0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_value = state_values[step] * (1 - is_dones[step])
                next_non_terminal = 1.0 - is_dones[step]

            delta = rewards[step] + self.gamma * next_value * next_non_terminal - state_values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[step] = gae

            returns[step] = state_values[step] + gae

            next_value = state_values[step]
            next_non_terminal = 1.0 - is_dones[step]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        mean_policy_loss = 0.0
        mean_value_loss = 0.0
        mean_entropy_loss = 0.0
        mean_total_loss = 0.0
        times = 0
        for _ in range(self.update_epochs):
            indices = torch.randperm(len(old_states))
            for start in range(0, len(old_states), 64):  # batch_size=64
                end = start + 64
                idx = indices[start:end]
                batch_states = old_states[idx]
                batch_actions = old_actions[idx]
                batch_old_logprobs = old_logprobs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]

                logprobs, dist_entropy, values = self.evaluate(batch_states, batch_actions)

                ratios = torch.exp(logprobs - batch_old_logprobs.detach())
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * self.MseLoss(values, batch_returns)

                entropy_loss = -0.01 * dist_entropy.mean()

                total_loss = policy_loss + value_loss + entropy_loss

                mean_policy_loss += policy_loss
                mean_value_loss += value_loss
                mean_entropy_loss += entropy_loss
                mean_total_loss += total_loss
                times +=1

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)  # 梯度裁剪
                self.optimizer.step()

        self.writer.add_scalar('Loss/policy_loss',  mean_policy_loss/times,     iteration)
        self.writer.add_scalar('Loss/value_loss',   mean_value_loss/times,      iteration)
        self.writer.add_scalar('Loss/entropy_loss', mean_entropy_loss/times,    iteration)
        self.writer.add_scalar('Loss/total_loss',   mean_total_loss/times,      iteration)

        self.scheduler.step()

        return

    def save(self, episode_reward):
        filename = f'export/model_{episode_reward:.0f}.pth'
        torch.save(
            {
                'model_state_dict': self.actor_critic.state_dict(),
                'reward': episode_reward
            }, 
            filename
        )

        return