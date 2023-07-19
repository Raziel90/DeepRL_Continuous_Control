from typing import List, Type

import numpy as np
import torch
from torch import nn



class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_sizes: List[int], action_limit: float=1.,
                 hidden_activation: Type[nn.Module] = nn.ReLU, output_activation: Type[nn.Module] = nn.Tanh,
                 random_seed=None) -> None:
        super(Actor, self).__init__()

        self.seed = None if random_seed is None else torch.manual_seed(random_seed)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.action_limit = action_limit
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layers = self.__build_net()

        self.reset()

    @classmethod
    def from_state(cls, state):
        args = {key : val for key, val in state.items()}
        instance = cls(**args)
        instance.set_state(state)
        return instance

    @classmethod
    def from_file(cls, file_path):
        state = torch.load(file_path)
        return cls.from_state(state)

    @property
    def network_dim(self):
        return np.sum([np.prod(param.shape) for param in self.parameters()])

    def reset(self):
        lim = 1 / np.sqrt(self.state_dim)
        self.layers[0].weight.data.uniform_(-lim, lim)
        for layer in self.layers[2:]:
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = layer(x)

        return x * self.action_limit

    def __build_net(self):
        layers_sizes = [self.state_dim] + self.hidden_sizes
        layers = []
        for fan_in, fan_out in zip(layers_sizes[:-1], layers_sizes[1:]):
            layers += [nn.Linear(fan_in, fan_out), self.hidden_activation()]
        layers += [nn.Linear(layers_sizes[-1], self.action_dim), self.output_activation()]

        return nn.ModuleList(layers)

    def get_state(self):
        network = {'params':  self.state_dict()}
        state = {
            'network': network,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_sizes': self.hidden_sizes,
            'action_limit': self.action_limit,
            'hidden_activation': self.hidden_activation,
            'output_activation': self.output_activation}
        return state

    def set_state(self, state):
        self.load_state_dict(state['network']['params'])
        self.state_dim = state['state_dim']
        self.action_dim = state['action_dim']
        self.hidden_sizes = state['hidden_sizes']
        self.action_limit = state['action_limit']
        self.hidden_activation = state['hidden_activation']
        self.output_activation = state['output_activation']

    def soft_update(self, other, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            other: PyTorch model (weights will be copied from)
            tau (float): interpolation parameter
        """
        for this_param, other_param in zip(self.parameters(), other.parameters()):
            this_param.data.copy_(tau * this_param.data + (1.0 - tau) * other_param.data)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int],
                 activation: Type[nn.Module] = nn.LeakyReLU,
                 random_seed=None) -> None:
        super(Critic, self).__init__()

        self.seed = None if random_seed is None else torch.manual_seed(random_seed)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.layers = self.__build_net()

        self.reset()

    @classmethod
    def from_state(cls, state):
        args = {key : val for key, val in state.items()}
        instance = cls(**args)
        instance.set_state(state)
        return instance

    @classmethod
    def from_file(cls, file_path):
        state = torch.load(file_path)
        return cls.from_state(state)

    @property
    def network_dim(self):
        return np.sum([np.prod(param.shape) for param in self.parameters()])

    def reset(self):
        for layer in self.layers[1:]:
            if isinstance(layer, nn.Linear):
                lim = 1 / np.sqrt(layer.weight.data.size()[0])
                layer.weight.data.uniform_(-lim, lim)

        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        fc, activation = self.layers[0], self.layers[1]
        x = activation(fc(state)) # apply first layer only on state
        x = torch.cat([x, action], dim=1) # apply transfer function and concatenate with action
        for layer in self.layers[2:]:
            x = layer(x)
        return x

    def __build_net(self):
        layers_sizes = [self.state_dim] + self.hidden_sizes

        layers = []
        for layer, (fan_in, fan_out) in enumerate(zip(layers_sizes[:-1], layers_sizes[1:])):
            fan_in += self.action_dim * (layer == 1)
            layers += [nn.Linear(fan_in, fan_out), self.activation()]
        layers += [nn.Linear(layers_sizes[-1], 1)]

        return nn.ModuleList(layers)


    def get_state(self):
        network = {'params':  self.state_dict()}
        state = {
            'network': network,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation}
        return state

    def set_state(self, state):
        self.load_state_dict(state['network']['params'])
        self.state_dim = state['state_dim']
        self.action_dim = state['action_dim']
        self.hidden_sizes = state['hidden_sizes']
        self.activation = state['activation']

    def soft_update(self, other, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            other: PyTorch model (weights will be copied from)
            tau (float): interpolation parameter
        """
        for this_param, other_param in zip(self.parameters(), other.parameters()):
            this_param.data.copy_(tau * this_param.data + (1.0 - tau) * other_param.data)

class DDPGPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 actor_hidden_sizes: List[int], critic_hidden_sizes: List[int],
                 action_limit: float = 1.,
                 hidden_actor_activation: Type[nn.Module] = nn.ReLU,
                 output_actor_activation: Type[nn.Module] = nn.Tanh,
                 hidden_critic_activation: Type[nn.Module] = nn.LeakyReLU,
                 seed=None) -> None:
        super(DDPGPolicy, self).__init__()

        torch.manual_seed(seed) if seed is not None else seed
        self.seed = seed

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit = action_limit
        self.actor_hidden_sizes = actor_hidden_sizes
        self.critic_hidden_sizes = critic_hidden_sizes

        self.pi = Actor(state_dim, action_dim, actor_hidden_sizes, action_limit,
                        hidden_actor_activation, output_actor_activation, seed)
        self.V = Critic(state_dim, action_dim, critic_hidden_sizes, hidden_critic_activation, seed)
    @property
    def network_dim(self):
        return self.pi.network_dim, self.V.network_dim

    @classmethod
    def from_state(cls, state):
        args = {key : val for key, val in state.items() if key not in ['pi', 'V']}
        args['actor_hidden_sizes'] = state['pi']['hidden_sizes']
        args['critic_hidden_sizes'] = state['V']['hidden_sizes']
        instance = cls(**args)
        instance.set_state(state)

        return instance

    @classmethod
    def from_file(cls, file_path):
        state = torch.load(file_path)
        return cls.from_state(state)

    def to_file(self, file_path):
        torch.save(self.get_state(), file_path)

    def reset(self):
        self.pi.reset()
        self.V.reset()

    def act(self, state):
        return self.pi(state)

    def evaluate(self, state, action):
        return self.V(state, action)

    def get_state(self):
        state = {
            'pi': self.pi.get_state(),
            'V': self.V.get_state(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'seed': self.seed
            }
        return state

    def set_state(self, state):
        self.pi.set_state(state['pi'])
        self.V.set_state(state['V'])
        self.state_dim = state['state_dim']
        self.action_dim = state['action_dim']

    def soft_update(self, other, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            other: PyTorch model (weights will be copied from)
            tau (float): interpolation parameter
        """
        self.pi.soft_update(other.pi, tau)
        self.V.soft_update(other.V, tau)

if __name__ == '__main__':
    policy = DDPGPolicy(4, 5, [128, 512], [128, 512])
    print(policy.pi.layers)
    print(policy.V.layers)
    policy.to_file('testDDPGPolicy.pth')

    policy2 = DDPGPolicy.from_file('testDDPGPolicy.pth')
    print(policy2.pi.layers)
    print(policy2.V.layers)
    print(policy2.pi.network_dim)
