
import numpy as np
import torch
import torch.optim as optim
from .model import Actor, Critic
from .replay_buffers import ReplayBuffer, PrioritizedReplayBuffer
import random



BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class DDPGAgent():

    def __init__(self, state_size, action_size, hidden_size=None, seed=0, **kwargs):


        self.seed = random.seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        self.actor_local = Actor(state_size, action_size, seed=self.seed, hidden_size=[24, 48]).to(device)
        self.actor_target = Actor(state_size, action_size, seed=self.seed, hidden_size=[24, 48]).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_size, action_size, seed=seed, hidden_size=[24, 48]).to(device)
        self.critic_target = Critic(state_size, action_size, seed=seed, hidden_size=[24, 48]).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.replay_buffer = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # self.hidden_layer_sizes = self.qnetwork_local.hidden_layer_sizes