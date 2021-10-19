
import numpy as np
import torch
import torch.nn.functional as F
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

        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # self.hidden_layer_sizes = self.qnetwork_local.hidden_layer_sizes

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, ):
        state = torch.from_numpy(state).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # if add_noise:
        #     action += self.noise.sample()
        action = (action + 1.0) / 2.0
        return np.clip(action, 0, 1)

    def learn(self, experiences, gamma):

        states, actions, rewards, next_states, dones = experiences

        # ------------------------------------------------------
        actions_next = self.actor_local(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
        Q_targets_expected = self.critic_local(states, actions)

        critic_loss = F.mse_loss(Q_targets_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ------------------------------------------------------
        action_predicted = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------------------------------------

        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)

        

    def soft_update(self, local_net, target_net, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_net: PyTorch model (weights will be copied from)
            target_net: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """

        for local_param, target_param in zip(local_net, target_net):
            target_param.data.copy(tau * local_param.data + (1.0 - tau) * target_param.data)