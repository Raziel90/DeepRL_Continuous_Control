import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class DDPGNet(nn.Module):
    def __init__(self, states, actions, hidden_layers=None, seed=None):
        super(DDPGNet, self).__init__()
        if hidden_layers is None:
            hidden_layers = [64, 64]
        if seed is None:
            self.seed = torch.seed()
        else:
            self.seed = torch.manual_seed(seed)
        
        self.n_states = states
        self.n_actions = actions

    def reset_parameters(self):

        self.input_layer.weight.data.uniform_(*self._init_layer(self.input_layer))
        for i in range(len(self.hidden_layers)):
            self.hidden_layers[i].weights.data.uniform_(*self._init_layer(self.hidden_layers[i]))
        self.output_layer.weight.data.uniform_(-3e-3, 3e-3)

    def _init_layer(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)
        
        

class Actor(nn.Module):
    
    def __init__(self, states, actions, hidden_layers=None, seed=None):
        super(Actor, self).__init__(states, actions, hidden_layers=None, seed=None)

        self.input_layer = nn.Linear(self.n_states, self.hidden_layer[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(l_sz_prev, l_sz) for l_sz_prev, l_sz in zip(self.hidden_layer[0:-1], self.hidden_layer[1:])
            ])
        self.output_layer = nn.Linear(self.hidden_layer[-1], self.n_actions)

        self.reset_parameters()

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.input_layer(state))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        return F.tanh(self.output_layer(x))

        

class Critic(nn.Module):
    """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hiddn_layers (list): List of integers corresponding to the sizes of hidden layers
            seed (int): Random seed
    """
    def __init__(self, states, actions, hidden_layers=None, seed=None):
        super(Critic, self).__init__(states, actions, hidden_layers=None, seed=None)

        self.input_layer = nn.Linear(self.n_states, self.hidden_layer[0])
        layer_list = [nn.Linear(self.hidden_layer[0] + actions, self.hidden_layer[1])]
        layer_list += [
            nn.Linear(l_sz_prev, l_sz) for l_sz_prev, l_sz in zip(self.hidden_layer[1:-1], self.hidden_layer[2:])
            ]
        self.hidden_layers = nn.ModuleList(layer_list)
        self.output_layer = nn.Linear(self.hidden_layer[-1], 1)

        self.reset_parameters()

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.input_layer(state))
        x = torch.cat((xs, action), dim=1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        return self.output_layer(x)