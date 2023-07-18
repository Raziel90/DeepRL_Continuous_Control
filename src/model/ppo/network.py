from typing import List, Type, Optional, Dict
import numpy as np
import torch
from torch import nn
from torch import distributions as dist
from torch.nn import functional as F


class Actor(nn.Module):
    """
    Network of the Actor of the PPO Agent
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_sizes: List[int], action_limit: float=1.,
                 hidden_activation: Type[nn.Module] = nn.ReLU, output_activation: Type[nn.Module] = nn.Tanh,
                 random_seed: Optional[int] = None) -> None:
        """Initializes the PPO Actor

        Parameters
        ----------
        state_dim : int
            dimentionality of the states
        action_dim : int
            dimentionality of the actions
        hidden_sizes : List[int]
            list of the number of neurons in each hidden layer
        action_limit : float, optional
            extremes of the action space, by default 1.
        hidden_activation : Type[nn.Module], optional
            activation function of the hidden layers, by default nn.ReLU
        output_activation : Type[nn.Module], optional
            activation function of the output layer (Actor only), by default nn.Tanh
        random_seed : _type_, optional
            random seed used, by default None
        """
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
    def from_state(cls, state: Dict):
        """Initializes an Actor network from the state dictionary

        Parameters
        ----------
        state : dict
            state dictionary of the actor net

        Returns
        -------
        initialised Actor
        """
        args = {key : val for key, val in state.items()}
        instance = cls(**args)
        instance.set_state(state)
        return instance

    @classmethod
    def from_file(cls, file_path: str):
        """initializes an Actor from a file

        Parameters
        ----------
        file_path : str
            path to the file

        Returns
        -------
        initialised Actor
        """

        state = torch.load(file_path)
        return cls.from_state(state)

    @property
    def network_dim(self):
        """Returns the number of trainable parameters in the network"""
        return np.sum([np.prod(param.shape) for param in self.parameters()])

    def reset(self):
        """Initialises the network"""
        for layer in self.layers[1:]:
            if isinstance(layer, nn.Linear):
                lim = 1 / np.sqrt(layer.weight.data.size()[0])
                layer.weight.data.uniform_(-lim, lim)

        self.layers[-2].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Implements the callable behaviour of the network"""
        x = state
        for layer in self.layers:
            x = layer(x)

        return x * self.action_limit

    def __build_net(self):
        """Composes the modules of the network"""
        layers_sizes = [self.state_dim] + self.hidden_sizes
        layers = []
        for fan_in, fan_out in zip(layers_sizes[:-1], layers_sizes[1:]):
            layers += [nn.Linear(fan_in, fan_out), self.hidden_activation()]
        layers += [nn.Linear(layers_sizes[-1], self.action_dim), self.output_activation()]

        return nn.ModuleList(layers)

    def get_state(self) -> Dict:
        """Returns the state dictionary of the network

        Returns
        -------
        state : dict
            state dictionary of the network
        """
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
        """
        Sets the state dictionary of the network

        Parameters
        ----------
        state : dict
            state dictionary of the network

        """
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
    """Critic (Value) Model."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int],
                 activation: Type[nn.Module] = nn.LeakyReLU,
                 random_seed: Optional[int] = None) -> None:
        """Initializes the Critic of the policy

       Parameters
        ----------
        state_dim : int
            dimentionality of the states
        action_dim : int
            dimentionality of the actions
        hidden_sizes : List[int]
            list of the number of neurons in each hidden layer
        action_limit : float, optional
            extremes of the action space, by default 1.
        activation : Type[nn.Module], optional
            _description_, by default nn.LeakyReLU
        random_seed : int, optional
            random seed used, by default None

        """
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
        """Initializes an Actor network from the state dictionary

        Parameters
        ----------
        state : dict
            state dictionary of the actor net

        Returns
        -------
        initialised Actor
        """
        args = {key : val for key, val in state.items()}
        instance = cls(**args)
        instance.set_state(state)
        return instance

    @classmethod
    def from_file(cls, file_path):
        """initializes an Actor from a file

        Parameters
        ----------
        file_path : str
            path to the file

        Returns
        -------
        initialised Actor
        """
        state = torch.load(file_path)
        return cls.from_state(state)

    @property
    def network_dim(self):
        """Returns the number of trainable parameters in the network"""
        return np.sum([np.prod(param.shape) for param in self.parameters()])

    def reset(self):
        """Initialises the network"""
        for layer in self.layers[1:]:
            if isinstance(layer, nn.Linear):
                lim = 1 / np.sqrt(layer.weight.data.size()[0])
                layer.weight.data.uniform_(-lim, lim)

        self.layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = layer(x)
        return x

    def __build_net(self):
        """Constructs the network"""
        layers_sizes = [self.state_dim] + self.hidden_sizes

        layers = []
        for layer, (fan_in, fan_out) in enumerate(zip(layers_sizes[:-1], layers_sizes[1:])):
            layers += [nn.Linear(fan_in, fan_out), self.activation()]
        layers += [nn.Linear(layers_sizes[-1], 1)]

        return nn.ModuleList(layers)


    def get_state(self):
        """Returns the state dictionary of the network

        Returns
        -------
        state : dict
            state dictionary of the network
        """
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


class PPOPolicy(nn.Module):
    """Actor-Critic PPO """
    def __init__(self, state_dim, action_dim,
                 actor_hidden_sizes: List[int], critic_hidden_sizes: List[int],
                 action_limit: float, initial_std_value: float= 1.,
                 hidden_activation: Type[nn.Module] = nn.ReLU, output_activation: Type[nn.Module] = nn.Tanh,
                 seed: Optional[int] = None):
        """Initializes the PPO policy

        Parameters
        ----------
        state_dim : int
            dimentionality of the states
        action_dim : int
            dimentionality of the actions
        actor_hidden_sizes : List[int]
            list of the number of neurons in each hidden layer
        critic_hidden_sizes : List[int]
            list of the number of neurons in each hidden layer
        action_limit : float
            extremes of the action space, by default 1.
        initial_std_value : float, optional
            initial values of the exploration standard deviation, by default 1.
        hidden_activation : Type[nn.Module], optional
            activation function for hidden units, by default nn.ReLU
        output_activation : Type[nn.Module], optional
            activation function used in the output units (OnlyActor), by default nn.Tanh
        seed : Optional[int], optional
            random seed, by default None
        """

        super(PPOPolicy, self).__init__()

        self.seed = torch.manual_seed(seed) if seed is not None else None
        self.state_dim, self.action_dim = state_dim, action_dim
        self.actor_hidden_sizes, self.critic_hidden_sizes = actor_hidden_sizes, critic_hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.action_limit = action_limit

        self.pi = Actor(state_dim, action_dim, actor_hidden_sizes, action_limit, hidden_activation, output_activation, seed)
        self.V = Critic(state_dim, action_dim, critic_hidden_sizes, hidden_activation, seed)
        self.log_std = nn.Parameter(torch.ones(1, self.action_dim) * torch.log(torch.as_tensor(initial_std_value)))

    def reset(self):
        """
        Initializes the policy.
        """
        self.pi.reset()
        self.V.reset()

    @classmethod
    def from_state(cls, state):
        """Loads the policy from the provided state dictionary"""
        args = {key : val for key, val in state.items() if key not in ['pi', 'V']}
        args['actor_hidden_sizes'] = state['pi']['hidden_sizes']
        args['critic_hidden_sizes'] = state['V']['hidden_sizes']
        instance = cls(**args)
        instance.set_state(state)

        return instance

    @classmethod
    def from_file(cls, file_path):
        """Loads the policy from the provided file path"""
        state = torch.load(file_path)
        return cls.from_state(state)

    def to_file(self, file_path):
        """Saves the policy to the provided file path"""

        torch.save(self.get_state(), file_path)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        action, value = self.pi(state), self.V(state)

        return action, value

    def state_value(self, state: Union[np.ndarray, torch.Tensor]):
        """Estimates the value of a state

        Parameters
        ----------
        state : np.ndarray or torch.Tensor
            state of the environment


        Returns
        -------
        float (n_threads X 1)
            value of the state (n_threads X 1)
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        return self.V(state)

    def action_prob(self, state, action):
        """Estimates the probability of an action in a state

        Parameters
        ----------
        state : np.ndarray or torch.Tensor
            state of the environment
        action : np.ndarray or torch.Tensor
            action to be estimated

        Returns
        -------
        float (n_threads X 1)
            probability of the action (n_threads X 1)
        """

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        estimated_action = self.pi(state)
        action_dist = dist.Normal(estimated_action, torch.exp(self.log_std))
        log_prob = action_dist.log_prob(action)

        return log_prob.sum(dim=1), action_dist.entropy().sum(dim=1)

    def act(self, state: Union[np.ndarray, torch.Tensor], noisy: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Acts in the environment, while observing a state

        Parameters
        ----------
        state : Union[np.ndarray, torch.Tensor]
            state of the environment
        noisy : bool, optional
            if true the action applied will be sampled from a Normal distribution of
            mean the output of the policy and std the exponential of the log_std of
            the class, by default False

        Returns
        -------
        Tuple
            the action chosen and it's log-probability
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        action = self.pi(state)
        action_dist = dist.Normal(action, torch.exp(self.log_std))
        if noisy:
            action = action_dist.sample()

        action = action.clamp(-self.action_limit, self.action_limit)
        log_prob = action_dist.log_prob(action).sum(dim=1)

        return action, log_prob

    def get_state(self) -> Dict:
        """Extracts the state of the policy to dictionary

        Returns
        -------
        Dict
            the state of the policy as a dictionary
        """
        state = {
            'pi': self.pi.get_state(),
            'V': self.V.get_state(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim}
        return state

    def set_state(self, state):
        """Assigns the state to the policy"""
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
    state_dim = 2
    action_dim = 2

    policy = PPOPolicy(state_dim, action_dim,
                      actor_hidden_sizes=[32, 32],
                      critic_hidden_sizes=[32, 32],
                      action_limit=1.0,
                      hidden_activation=nn.ReLU,
                      output_activation=nn.Tanh)

    print(policy.pi)
    print(policy.V)
    print(policy.log_std)
