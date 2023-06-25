import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import cat
from .nn_bodies import BodyBase

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeterministicActorCritic(nn.Module):
    def __init__(self, 
                 state_dim, action_dim,
                 actor_optimizer, critic_optimizer,
                 actor_lr, critic_lr,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        
        super().__init__()

        self.phi_body = BodyBase(state_dim=state_dim) if phi_body is None else phi_body
        self.actor_body = BodyBase(phi_body.feature_dim) if actor_body is None else actor_body
        self.critic_body = BodyBase(phi_body.feature_dim) if critic_body is None else critic_body

        self.fc_actor = nn.Linear(actor_body.feature_dim, action_dim)
        self.fc_critic = nn.Linear(critic_body.feature_dim, 1)

        self.actor_params = self.actor_body.parameters() + self.fc_actor.parameters()
        self.critic_params = self.critic_body.parameters() + self.fc_critic.parameters()
        self.phi_params = self.phi_body.parameters()

        self.actor_opt = actor_optimizer(self.actor_params + self.phi_params, actor_lr)
        self.critic_opt = critic_optimizer(self.actor_params + self.phi_params, critic_lr)

        self.init_parameters(1e-3)

        self.to(device)

    def init_parameters(self, w_scale):

        for layer in [self.fc_actor, self.fc_critic]:
            nn.init.orthogonal_(layer, layer.weight.data)
            layer.weight.data.mul_(w_scale)
            nn.init.constant_(layer.bias.data, 0)
    
    def forward(self, x):
        phi = self.feature(x)
        
        return self.actor(phi)
    
    def actor(self, phi):
        return F.tanh(
            self.fc_actor(self.actor_body(phi))
            )
    
    def critic(self, phi, a):
        return (
            self.fc_critic(self.critic_body(torch.cat([phi, a], dim=1)))
            )

