import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_mlp(layer_sizes, activation, output_activation=nn.ReLU):
    layers = []
    for j, (size_in, size_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_activation = activation if j < len(layer_sizes) else output_activation
        layers += [nn.Linear(size_in, size_out), layer_activation()]
    
    return nn.Sequential(*layers)

def network_dim(nn_module):
    return sum([np.prod(p.shape) for p in nn_module.parameters()])


class Actor_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, action_limit):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit = action_limit

        pi_sizes = [state_dim] + list(hidden_sizes) + [action_dim]
        self.pi = generate_mlp(pi_sizes, activation, nn.Tanh)

    def forward(self, state):
        return self.action_limit * self.pi(state)


class Critic_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q = generate_mlp([state_dim + action_dim] + list(hidden_sizes) + [1], activation)
    
    def forward(self, state, action):
        state_space = torch.cat([state, action], dim=-1)
        q = self.q(state_space)
        return torch.squeeze(q, -1)



class ActorCritic_Net(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.hidden_sizes = hidden_sizes

        self.pi = Actor_Net(state_dim, action_dim, hidden_sizes, activation, action_limit)
        self.q = Critic_Net(state_dim, action_dim, hidden_sizes, activation)

    def act(self, state):
        with torch.no_grad():
            action = self.pi(state)
            return action.numpy()
        
    @classmethod
    def from_file(cls, filename):
        """Builds the Agent from file

        Args:
            data (dict): dict representation of the agent

        Returns:
            Agent: agent object correspondent to the parameters
        """
        data = torch.load(filename)
        obj = cls.from_dict(data)
        return obj
    
    @classmethod
    def from_dict(cls, data):
        """Builds the Agent from a dict representation of the agent

        Args:
            data (dict): dict representation of the agent

        Returns:
            Agent: agent object correspondent to the parameters
        """
        obj = cls(data['state_size'], data['action_size'], data['action_limit'], data['hidden_layer_sizes'], **data['kwargs'])
        obj.load_state_dict(data['weights'])
        return obj

    def to_dict(self):
        """Extrapolates agent's parameter and hyper-parameters and organises them in a dict object

        Returns:
            dict: dict representation of the agent
        """
        data = {
            'state_size': self.state_dim,
            'action_size': self.action_dim,
            'action_limit': self.action_limit,
            'hidden_layer_sizes': self.hidden_sizes,
            'weights': self.state_dict(),
            'kwargs': {}
        }
        return data

    def save_model(self, filename):
        """Saves the agent on a file

        Args:
            filename (str): path to the file to save
        """
        data = self.to_dict()
        torch.save(data, filename)