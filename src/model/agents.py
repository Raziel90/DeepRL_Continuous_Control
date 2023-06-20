import numpy as np
from copy import deepcopy
from collections import deque
import torch
import logging
import random
from unityagents import UnityEnvironment

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .actor_critic_net import ActorCritic_Net
from .replay_buffers.buffer_replay import PrioritizedReplayBuffer



LOGGER = logging.getLogger(__name__)
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():

    def __init__(self, 
                #  state_size, action_size, 
                env : UnityEnvironment,
                hidden_sizes=None, seed=None, **kwargs):
        raise NotImplementedError
        
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
        obj.qnetwork_local.load_state_dict(data['weights'])
        return obj
    
    @classmethod
    def from_dict(cls, data):
        """Builds the Agent from a dict representation of the agent

        Args:
            data (dict): dict representation of the agent

        Returns:
            Agent: agent object correspondent to the parameters
        """
        return cls(data['state_size'], data['action_size'], data['hidden_layer_sizes'], **data['kwargs'])

    def to_dict(self):
        """Extrapolates agent's parameter and hyper-parameters and organises them in a dict object

        Returns:
            dict: dict representation of the agent
        """
        data = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'weights': self.qnetwork_local.state_dict(),
            'seed': self.qnetwork_local.seed.seed(),
            'kwargs': {'dueling': self.dueling_networks}
        }
        return data

    def save_model(self, filename):
        """Saves the agent on a file

        Args:
            filename (str): path to the file to save
        """
        data = self.to_dict()
        torch.save(data, filename)


TAU = 1e-3              # for soft update of target parameters
LR = 1e-3               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
PRIORITY_PROBABILITY_A = .9 # Coefficient used to compute the importance of the priority weights during buffer sampling
PRIORITY_CORRECTION_B = 1. # Corrective factor for the loss in case of Priority Replay Buffer

ACTION_LIMIT = np.pi
HIDDEN_UNITS = (256, 256)
ACTIVATION = nn.ReLU

class DDPG_Agent(Agent):
    def __init__(
            self, env: UnityEnvironment,
            # state_size, action_size, 
            network_config=None, seed=None, **kwargs):
        
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        self.env = env
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.state_size = env_info.vector_observations.shape[1]
        self.action_size = self.brain.vector_action_space_size

        network_config = dict() if network_config is None else network_config
        self.action_limit = network_config.get('action_limit', ACTION_LIMIT)
        hidden_sizes = network_config.get('hidden_sizes', HIDDEN_UNITS)
        activation = network_config.get('activation', ACTIVATION)
        self.buffer_size = network_config.get('buffer_size', BUFFER_SIZE)
        self.batch_size = network_config.get('batch_size', BATCH_SIZE)

        self.gamma = kwargs.get('gamma', GAMMA)
        self.update_every = kwargs.get('update_every', UPDATE_EVERY)
        self.buffer_size = kwargs.get('buffer_size', BUFFER_SIZE)
        self.tau = kwargs.get('tau', TAU)

        self.pi_lr = kwargs.get('pi_lr', LR)
        self.q_lr = kwargs.get('q_lr', LR)

        self.actor_critic = ActorCritic_Net(
            self.state_size, self.action_size, 
            self.action_limit, 
            hidden_sizes=hidden_sizes,
            activation=activation)
        
        self.pi_optimizer = optim.Adam(self.actor_critic.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = optim.Adam(self.actor_critic.q.parameters(), lr=self.q_lr)

        # Create target newtwork and freeze the weight to optimizers
        self.target_actor_critic = deepcopy(self.actor_critic)
        for p in self.target_actor_critic.parameters():
            p.requires_grad = False

        self.memory = PrioritizedReplayBuffer(self.action_size, self.buffer_size, self.batch_size, seed)

        LOGGER.info('-' * 20)
        LOGGER.info(f' State dimension: {self.state_size}\t Action dimension: {self.action_size}\t Action Limit: {self.action_limit}')
        LOGGER.info(f'Network hidden units: {hidden_sizes} -> Total hidden weights: {np.prod(hidden_sizes)}')
        LOGGER.info('Network params: ')
        LOGGER.info(str(network_config))
        LOGGER.info('-' * 20)

    def soft_update(self):
        """Soft update of the model parameters. It applies a linear interpolation based on the self.tau parameter
           target_θ = τ × local_θ + (1 - τ) × target_θ
        """
        local_model, target_model = self.actor_critic, self.target_actor_critic

        with torch.no_grad():
            for local_par, target_par in zip(local_model.pi.parameters(), target_model.pi.parameters()):
                target_par.data.copy_(self.tau * local_par.data + (1-self.tau) * target_par.data)

    def act(self, state, eps=0.):
        """Performs an agent action

        Args:
            state (array of floats): current state vector
            eps (float, optional): ϵ for ϵ-greedy agent. Indicates the probability of a random action. Defaults to 0..

        Returns:
            [type]: [description]
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def get_action(self, state, noise_scale=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        act = self.actor_critic.act(state)
        act += noise_scale * np.random.random(self.action_size)
        return np.clip(act, -self.action_limit, self.action_limit)

    def run_training(self, n_episodes, max_duration, eps_start=0.1, eps_end=0.01, eps_decay=0.9990, update_every=50, update_after=10000):

        env_info = self.env.reset(train_mode=True)[self.brain_name] # reset the environment
        state = env_info.vector_observations[0].astype(float)   
        score = 0
        eps = eps_start
        max_average_score = 0.
        scores_window = deque(maxlen=100) 
        num_steps = 0
        for e in range(n_episodes):
            episode_score = 0
            eps = max(eps_end, eps_decay * eps) # decrease epsilon
            for t in range(max_duration):
                action = self.get_action(state, eps)
                env_info = self.env.step(action)[self.brain_name]        # send the action to the environment
                next_state = env_info.vector_observations[0].astype(float)   # get the next state
                reward = env_info.rewards[0]                   # get the reward
                done = env_info.local_done[0]
                self.memory.add(state, action, reward, next_state, done)
                state = next_state
                
                episode_score += reward
                num_steps += 1

                if num_steps > update_after and num_steps % update_every == 0:
                    # LOGGER.info('\rUpdating network')
                    batch = self.memory.sample()
                    self.update(batch)

            scores_window.append(episode_score)  

            if e % 100 ==0:
                # Save Agent
                LOGGER.info(f'\rEpisode {e}: {num_steps} steps\tAverage Score: {np.mean(scores_window):.2f}')
                self.actor_critic.save_model('ddpg.pth')
                # Test agent
                pass

    def update(self, batch):
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(batch)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze the Q parameters while computing the PI gradients
        for param in self.actor_critic.q.parameters():
            param.requires_grad = False

        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(batch)
        loss_pi.backward()
        self.pi_optimizer.step()

        for param in self.actor_critic.q.parameters():
            param.requires_grad = True

        self.soft_update()
    
    def compute_loss_q(self, batch):


        (state, action, reward, next_state, done, _) = batch
        q = self.actor_critic.q(state, action)

        with torch.no_grad():
            q_pi_target = self.target_actor_critic.q(
                next_state, self.target_actor_critic.pi(next_state))
            backup = reward + self.gamma * (1 - done) * q_pi_target

        loss_q = (((q - backup)) ** 2).mean()

        return loss_q

    def compute_loss_pi(self, batch):
        (state, action, reward, next_state, done, _) = batch
        return - self.actor_critic.q(state, self.actor_critic.pi(state)).mean()
                    

    def test_agent(self, num_test_episodes, max_ep_len):
        for j in range(num_test_episodes):
            env_info = self.env.reset(train_mode=False)[self.brain_name] # reset the environment
            state = env_info.vector_observations[0]
            done = False
            while not(done or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                # o, r, d, _ = self.env.step(self.get_action(o, 0))
                action = self.get_action(state, 0)
                env_info = self.env.step(action)[self.brain_name]        # send the action to the environment
                state = env_info.vector_observations[0]   # get the next state
                reward = env_info.rewards[0]                   # get the reward
                done = env_info.local_done[0]
                ep_ret += reward
                ep_len += 1
            LOGGER.info(TestEpRet=ep_ret, TestEpLen=ep_len)
