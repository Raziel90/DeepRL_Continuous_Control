import torch
from torch import nn
from typing import List, Type
import numpy as np

class Policy(nn.Module):


    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int], action_limit: float, action_std: float=1e-2,
                 hidden_activation: Type[nn.Module] = nn.ReLU, output_activation: Type[nn.Module] = nn.Tanh):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.action_limit = action_limit
        self.action_std = action_std
        self.set_action_var(self.action_std)

        self.stub = self.__create_network_stub()
        self.actor_head, self.critic_head = self.__create_network_heads()

        self.pi = nn.Sequential(self.stub, self.actor_head)
        self.V = nn.Sequential(self.stub, self.critic_head)
    
    @property
    def action_std(self):
        return self.__action_std
    
    @action_std.setter
    def action_std(self, action_std):
        self.__action_std = action_std
        self.set_action_var(action_std)

    def __create_network_stub(self):
        layer_sizes = [self.state_dim] + self.hidden_sizes# + [self.action_dim]
        layers = []

        for j, (size_in, size_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers += [nn.Linear(size_in, size_out), self.hidden_activation()]

        return nn.Sequential(*layers)
    
    def __create_network_heads(self):

        mean = nn.Sequential(nn.Linear(self.hidden_sizes[-1], self.action_dim), self.output_activation())
        # std = nn.Sequential(nn.Linear(self.hidden_sizes[-1], self.action_dim), nn.Softplus())
        value = nn.Sequential(nn.Linear(self.hidden_sizes[-1], 1), nn.ReLU())
        
        return mean, value
    

    def __create_networks(self):
        actor_layer_sizes = [self.state_dim] + self.hidden_sizes + [self.action_dim]
        critic_layer_sizes = [self.state_dim] + self.hidden_sizes + [1]
        actor_layers = []
        critic_layers = []

        for j, (size_in, size_out) in enumerate(zip(actor_layer_sizes[:-1], actor_layer_sizes[1:])):
            actor_layers += [
                nn.Linear(size_in, size_out), self.hidden_activation() if j < len(actor_layer_sizes) else self.output_activation()]
        
        for j, (size_in, size_out) in enumerate(zip(critic_layer_sizes[:-1], critic_layer_sizes[1:])):
            critic_layers += [nn.Linear(size_in, size_out), 
                                self.hidden_activation() if j < len(critic_layer_sizes) else nn.ReLU()]

        return nn.Sequential(*layers)
    
    @property
    def network_dim(self):
        return sum([np.prod(p.shape) for p in self.parameters()])
    
    def set_action_var(self, action_std):
        self.action_var = torch.full((self.action_dim,), action_std * action_std)

    def act(self, state: torch.Tensor):
        bottleneck = self.stub(state)
        mean = self.action_limit * self.actor_head(bottleneck)
        state_value = self.critic_head(bottleneck)
        act_dist = torch.distributions.MultivariateNormal(mean, torch.diag(self.action_var))
        action = act_dist.sample()
        return action.detach(), act_dist.log_prob(action).detach(), state_value.detach() 

    def evaluate(self, state, action):
        action_mean = self.pi(state)
        cov_mat = torch.diag(self.action_var)
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.V(state)

        return action_log_probs, state_values, dist_entropy

    def forward(self, state: torch.Tensor):

        return self.act(state)
    
    def set_state(self, state):
        network = state['network']
        self.stub.load_state_dict(network['stub'])
        self.actor_head.load_state_dict(network['actor_head'])
        self.critic_head.load_state_dict(network['critic_head'])

        self.action_std = state['action_std']
        self.action_dim = state['action_dim']
        self.state_dim = state['state_dim']
        self.hidden_sizes = state['hidden_sizes']
        self.action_limit = state['action_limit']
        self.hidden_activation = state['hidden_activation']
        self.output_activation = state['output_activation']

        self.set_action_var(self.action_std)


    def get_state(self):
        return {'network':
            {
                'stub': self.stub.state_dict(),
                'actor_head': self.actor_head.state_dict(),
                'critic_head': self.critic_head.state_dict()},
            'action_std': self.action_std,
            'action_dim': self.action_dim,
            'hidden_sizes': self.hidden_sizes,
            'state_dim': self.state_dim,
            'action_limit': self.action_limit,
            'hidden_activation': self.hidden_activation,
            'output_activation': self.output_activation
            }

    def save_model(self, path):
        state = self.get_state()
        torch.save(state, path)

    def load_model(self, path):
        state = torch.load(path)
        self.set_state(state)

if __name__ == '__main__':

    state_prototype = torch.from_numpy(np.random.randn(6)).float()
    policy = Policy(6, 4, [256, 256], 1.)
    print(state_prototype.size())
    print(policy.pi)
    print(policy.V)
    print(policy(state_prototype))