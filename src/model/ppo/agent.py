
from typing import Dict, Tuple
from unityagents import UnityEnvironment
import torch
from torch import nn
import numpy as np
import logging
from network import Policy


DEFAULT_HIDDEN_SIZES = [256, 256]
DEFAULT_ACTION_SCALE = 1.
DEFAULT_HIDDEN_ACTIVATION = nn.ReLU
DEFAULT_OUTPUT_ACTIVATION = nn.Tanh
DEFAULT_ACTION_STD_INIT = 1e-2

LOGGER = logging.getLogger(__name__)

class PPOAgent():

    def __init__(self, env: UnityEnvironment, action_std_min=1e-6,
            network_config: Dict=None, seed=None):
        if seed is not None:
            torch.seed(seed)
        self.env = env
        self.__brain_name = env.brain_names[0]
        self.__brain = env.brains[self.__brain_name]
        env_info = self.env.reset(train_mode=True)[self.__brain_name]
        self.state_size = env_info.vector_observations.shape[1]
        self.action_size = self.__brain.vector_action_space_size
        self.action_std_min = action_std_min

        network_config = network_config if network_config is not None else dict()
        network_config.setdefault('hidden_sizes', DEFAULT_HIDDEN_SIZES)
        network_config.setdefault('action_limit', DEFAULT_ACTION_SCALE)
        network_config.setdefault('hidden_activation', DEFAULT_HIDDEN_ACTIVATION)
        network_config.setdefault('output_activation', DEFAULT_OUTPUT_ACTIVATION)
        network_config.setdefault('action_std', DEFAULT_ACTION_STD_INIT)

        self.policy = Policy(self.state_size, self.action_size, **network_config)

        LOGGER.info('\t' + '-' * 31 + ' Params ' + '-' * 31)
        LOGGER.info(f'\tState dimension: {self.state_size}\t Action dimension: {self.action_size}\t Action Limit: {self.policy.action_limit}')
        LOGGER.info(f'\tNetwork hidden units: {network_config.get("hidden_sizes")} -> Total hidden weights: {np.prod(network_config.get("hidden_sizes"))}')
        LOGGER.info(f'\tNetwork hidden activations: {network_config.get("hidden_activation")}')
        LOGGER.info(f'\tNetwork output activation: {network_config.get("output_activation")}')
        LOGGER.info('\t' + '-' * 70)

    @classmethod
    def from_file(cls, env: UnityEnvironment, network_state_path:str, action_std_min=1e-6, seed=None):
        
        policy_config = torch.load(network_state_path)
        network_config = {key: item for key, item in policy_config.items() if key != 'network'}
            
        agent = cls(env, action_std_min, network_config, seed)
        agent.policy.set_state(policy_config)
        return agent
    
    def save_agent(self, path: str):
        self.policy.save_model(path=path)
    
    def get_action(self, state: torch.Tensor, random: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if random == False:
            action, log_prob, _ = self.policy.act(state)
        else:
            ones = torch.ones(self.action_size)
            distro = torch.distributions.Uniform(-self.policy.action_limit * ones, self.policy.action_limit * ones)
            action = distro.sample()
            log_prob = distro.log_prob(action).sum(dim=1)
        return action, log_prob
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.policy.V(state)
    
    def decay_action_std(self, decay_rate: float):
        self.policy.action_std = max(self.action_std_min, self.policy.action_std * decay_rate)

    def act(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clipped_action = np.clip(action.numpy(), -self.policy.action_limit, self.policy.action_limit)
        env_info = self.env.step(clipped_action)[self.__brain_name] 
        new_state = torch.from_numpy(env_info.vector_observations).float()   # get the next state
        reward = torch.tensor(env_info.rewards, dtype=torch.float).unsqueeze(0)                   # get the reward
        done = torch.tensor([env_info.local_done], dtype=torch.float)
        
        return new_state, reward, done

    def run_episode(self, max_steps: int, random_policy: bool = False, train_mode: bool=False):
        
        env_info = self.env.reset(train_mode=train_mode)[self.__brain_name] # reset the environment
        state = torch.from_numpy(env_info.vector_observations).float()
        done = False
        values = []
        states = []
        actions = []
        rewards = []
        dones = []
        for t in range(max_steps):

            action, log_prob = self.get_action(state, random=random_policy)
            value = self.get_value(state)
            new_state, reward, done = self.act(action)
            # env_info = self.env.step(action.numpy())[self.__brain_name]        # send the action to the environment

            # new_state = env_info.vector_observations[0]   # get the next state
            # reward = env_info.rewards[0]                   # get the reward
            # done = env_info.local_done[0]
            # trajectory.append((state, action, new_state, reward, values))

            values.append(value)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = new_state
            if done: 
                break            
        
        return values, states, actions, rewards, dones
    



if __name__ == "__main__":
    env = UnityEnvironment(file_name='/Users/claudiocoppola/code/RL_repos/Reacher.app', no_graphics=True)
    agent = PPOAgent(env)
    values, states, actions, rewards, masks = agent.run_episode(100)
    print(sum(rewards))