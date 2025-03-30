
import torch

kStateDim = 8
kActionDim = 4

class ActorCritic(torch.nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()

        self.actor_layers = torch.nn.Sequential(
            torch.nn.Linear(kStateDim, kStateDim),
            torch.nn.ReLU(),
            torch.nn.Linear(kStateDim, kStateDim),
            torch.nn.ReLU(),
            torch.nn.Linear(kStateDim, kActionDim),
            torch.nn.Softmax(dim=-1)
        )

        self.critic_layers = torch.nn.Sequential(
            torch.nn.Linear(kStateDim, kStateDim),
            torch.nn.ReLU(),
            torch.nn.Linear(kStateDim, kStateDim),
            torch.nn.ReLU(),
            torch.nn.Linear(kStateDim, 1)
        )
        return

