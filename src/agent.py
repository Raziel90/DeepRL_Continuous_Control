
import numpy as np
import torch
import torch.optim as optim
from .model import Actor, Critic
# from .replay_buffers import ReplayBuffer, PrioritizedReplayBuffer
import random




class DDPGAgent():

    def __init__(self, state_size, action_size, hidden_size=None, seed=0, **kwargs):


        self.state_size = state_size
        self.action_size = action_size

        self.actor_local = None
        self.actor_target = None

        self.critic_local = None
        self.critic_target = None
        
        # self.hidden_layer_sizes = self.qnetwork_local.hidden_layer_sizes