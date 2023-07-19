
import logging
from typing import Dict, Tuple, Optional, TypeVar, Union

from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from unityagents import UnityEnvironment
from .network import DDPGPolicy
from .buffer_replay import PrioritizedReplayBuffer

DEFAULT_HIDDEN_SIZES = [256, 256]
DEFAULT_ACTION_SCALE = 1.
DEFAULT_HIDDEN_ACTIVATION = nn.ReLU
DEFAULT_OUTPUT_ACTIVATION = nn.Tanh
DEFAULT_ACTION_STD_INIT = 1e-2

LOGGER = logging.getLogger(__name__)

GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay

TDDPGAgent = TypeVar("TDDPGAgent", bound="DDPGAgent")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    def __init__(self, env: UnityEnvironment, network_config: Optional[Dict]=None, seed=None):


        self.env = env
        self.__brain_name = env.brain_names[0]
        self.__brain = env.brains[self.__brain_name]
        env_info = self.env.reset(train_mode=True)[self.__brain_name]

        self.n_threads = len(env_info.agents)
        self.state_size = env_info.vector_observations.shape[1]
        self.action_size = self.__brain.vector_action_space_size
        # self.action_std_min = action_std_min
        self.seed = seed
        network_config = network_config if network_config is not None else dict()

        network_config.setdefault('action_limit', DEFAULT_ACTION_SCALE)
        network_config.setdefault('actor_hidden_sizes', DEFAULT_HIDDEN_SIZES)
        network_config.setdefault('critic_hidden_sizes', DEFAULT_HIDDEN_SIZES)
        network_config.setdefault('seed', self.seed)

        self.policy = DDPGPolicy(self.state_size, self.action_size, **network_config).to(device)

        self.noise = OUNoise(self.action_size, seed=seed)

    @classmethod
    def clone_agent(cls, agent: TDDPGAgent):

        network_config = {
            'action_limit': agent.policy.action_limit,
            'actor_hidden_sizes': agent.policy.actor_hidden_sizes,
            'critic_hidden_sizes': agent.policy.critic_hidden_sizes,
            'seed': agent.seed
            }

        new_agent = cls(agent.env, network_config=network_config, seed=agent.seed)
        new_agent.policy.load_state_dict(agent.policy.state_dict())
        return new_agent

    @classmethod
    def from_file(cls, env: UnityEnvironment, filename: str):
        state = torch.load(filename)
        # print(state['pi']['hidden_activation'])
        network_config = {
            'action_limit': state['pi']['action_limit'],
            'actor_hidden_sizes': state['pi']['hidden_sizes'],
            'critic_hidden_sizes': state['V']['hidden_sizes'],
            'hidden_actor_activation': state['pi']['hidden_activation'],
            'hidden_critic_activation': state['V']['activation'],
            'output_actor_activation': state['pi']['output_activation']
            }
        # print(network_config.keys())
        agent = cls(env, network_config=network_config, seed=state.get('seed', None))
        agent.policy.set_state(state)

        return agent

    def to_file(self, filename: str):
        torch.save(self.policy.get_state(), filename)

    def act(self, state: np.ndarray, add_noise: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.policy.eval()
        with torch.no_grad():
            action = self.policy.act(torch.from_numpy(np.asarray(state)).float())
        self.policy.train()

        if add_noise:
            action_noise = self.noise.sample().squeeze()
            action += torch.from_numpy(action_noise).float().to(device)

        clipped_action = np.clip(action.numpy(), -self.policy.action_limit, self.policy.action_limit)
        env_info = self.env.step(clipped_action)[self.__brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done

        return action, next_state, reward, done

    def noise_reset(self):
        self.noise.reset()

    def run_test_episode(self, max_steps: int = 1000, track_progress: bool = False) -> np.ndarray:
        """Runs a test episode."""
        env_info = self.env.reset(train_mode=False)[self.__brain_name]

        state = torch.from_numpy(env_info.vector_observations).float().to(device)
        episode_reward = np.zeros((1, self.n_threads))
        if track_progress:
            pbar = tqdm(range(max_steps))
        else:
            pbar = range(max_steps)
        for _ in pbar:
            action, state, reward, done = self.act(state, add_noise=False)
            episode_reward += np.asarray(reward)
            if track_progress:
                pbar.set_description(f"Mean Cumulative Reward: {episode_reward.mean():.2f}")
            if np.asarray(done).astype(bool).any():
                break

        return episode_reward


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size: int, mu: float = 0., theta: float = 0.15, sigma: float = 0.2, seed: Optional[int] = None):
        """Initialize parameters and noise process."""

        self.seed = None if seed is None else np.random.seed(seed)

        self.mu = mu * np.ones((1, size))
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self) -> None:
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self, n=1) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state

        # print(x.shape)
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(n, x.shape[1])
        self.state = x + dx
        return self.state


if __name__ == '__main__':

    # noise = OUNoise(5)
    # print(np.asarray([noise.sample() for _ in range(3)]))

    env = UnityEnvironment(
        file_name='/Users/claudcop/code/deep-rl/DeepRL_Continuous_Control/unity/Reacher20.app',
        no_graphics=False)

    agent = DDPGAgent(env)
    agent.run_test_episode(max_steps=1000, track_progress=True)
