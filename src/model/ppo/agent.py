
import torch
from torch import nn
from torch import distributions as dist
from typing import Dict, TypeVar
from unityagents import UnityEnvironment
import numpy as np
from scipy.stats import norm
from tqdm import trange

from .network import PPOPolicy

DEFAULT_HIDDEN_SIZES = [256, 128]
DEFAULT_ACTION_SCALE = 1.
DEFAULT_HIDDEN_ACTIVATION = nn.ReLU
DEFAULT_OUTPUT_ACTIVATION = nn.Tanh

PPOAgent_Type = TypeVar("PPOAgent_Type", bound="PPOAgent")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent:

    def __init__(self, env: UnityEnvironment,
            network_config: Dict=None, seed=None) -> None:

        self.seed = np.random.seed(seed) if seed is not None else None


        self.env = env
        self.__brain_name = env.brain_names[0]
        self.__brain = env.brains[self.__brain_name]
        env_info = self.env.reset(train_mode=True)[self.__brain_name]

        self.n_threads = len(env_info.agents)
        self.state_size = env_info.vector_observations.shape[1]
        self.action_size = self.__brain.vector_action_space_size

        network_config = network_config if network_config is not None else dict()

        network_config.setdefault('actor_hidden_sizes', DEFAULT_HIDDEN_SIZES)
        network_config.setdefault('critic_hidden_sizes', DEFAULT_HIDDEN_SIZES)
        network_config.setdefault('action_limit', DEFAULT_ACTION_SCALE)
        network_config.setdefault('hidden_activation', DEFAULT_HIDDEN_ACTIVATION)
        network_config.setdefault('output_activation', DEFAULT_OUTPUT_ACTIVATION)

        self.policy = PPOPolicy(self.state_size, self.action_size, seed=seed, **network_config)

    @classmethod
    def clone_agent(cls, agent: PPOAgent_Type):

        network_config = {
            'action_limit': agent.policy.action_limit,
            'actor_hidden_sizes': agent.policy.actor_hidden_sizes,
            'critic_hidden_sizes': agent.policy.critic_hidden_sizes,
            'hidden_activation': agent.policy.hidden_activation,
            'output_activation': agent.policy.output_activation,
            'seed': agent.seed
            }

        new_agent = cls(agent.env, network_config=network_config, seed=agent.seed)
        new_agent.policy.load_state_dict(agent.policy.state_dict())
        return new_agent

    def act(self, state: np.ndarray, noisy: bool=False) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        action, log_prob = self.policy.act(state, noisy)
        env_info = self.env.step(action.cpu().detach().numpy())[self.__brain_name]
        next_state = np.asarray(env_info.vector_observations)
        reward = np.asarray(env_info.rewards)[:, None]
        done = np.asarray(env_info.local_done).astype(np.int32)[:, None]

        return action, log_prob, next_state, reward, done

    def random_act(self, state: Union[np.ndarray, torch.Tensor]):
        action = np.random.uniform(low=-self.policy.action_limit, high=self.policy.action_limit, size=(self.n_threads, self.action_size))
        log_prob = (
            norm(np.zeros((self.n_threads, self.action_size)), np.ones((self.n_threads, self.action_size)))
            .logpdf(action)).sum(axis=1)
        action_distr = dist.Normal(torch.from_numpy(action).float(), self.policy.log_std.exp().detach())
        action = torch.clamp(action_distr.sample(), -self.policy.action_limit, self.policy.action_limit).detach()

        env_info = self.env.step(action.numpy())[self.__brain_name]
        log_prob = action_distr.log_prob(action).sum(dim=1).detach()

        next_state = np.asarray(env_info.vector_observations)
        reward = np.asarray(env_info.rewards)[:, None]
        done = np.asarray(env_info.local_done).astype(np.int32)[:, None]

        return action, log_prob, next_state, reward, done

    def run_test_episode(self, max_steps: int = 1000, track_progress: bool = False, noisy: bool=False) -> np.ndarray:
        """Runs a test episode."""
        rewards = []
        env_info = self.env.reset(train_mode=False)[self.__brain_name] # reset the environment
        state = np.asarray(env_info.vector_observations)
        pbar = trange(max_steps) if track_progress else range(max_steps)
        for _ in pbar:
            action, log_prob, next_state, reward, done = self.act(state, noisy)
            value = self.policy.state_value(state).detach().numpy()
            state = next_state
            rewards.append(reward.squeeze())
            if track_progress:
                pbar.set_description(f"Mean Cumulative Reward: {np.sum(rewards, axis=0).mean():.2f}")
            if np.array(done).astype(bool).any():
                break

        return np.array(rewards).sum(axis=0)


if __name__ == '__main__':

    env = UnityEnvironment(file_name='/Users/claudcop/code/deep-rl/DeepRL_Continuous_Control/unity/Reacher20.app', no_graphics=False)

    agent = PPOAgent(env, seed=4)
    agent.run_test_episode(max_steps=1000, track_progress=True)
    print('END!')
