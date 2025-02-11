
import torch

kStateDim = 8
kActionDim = 4

class ActorCritic(torch.nn.Module):
    def __init__(self, load_model):
        super(ActorCritic, self).__init__()

        self.actor_layers = torch.nn.Sequential(
            torch.nn.Linear(kStateDim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, kActionDim),
            torch.nn.Softmax(dim=-1)
        )

        self.critic_layers = torch.nn.Sequential(
            torch.nn.Linear(kStateDim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

        if load_model:
            self.actor_layers.load_state_dict(torch.load('actor_layers.pth'))
            self.critic_layers.load_state_dict(torch.load('critic_layers.pth'))
            print("model loaded")

