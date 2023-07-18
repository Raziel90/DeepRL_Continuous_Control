import numpy as np
import torch
from torch import nn
import torch
from dataclasses import dataclass, field
from typing import Tuple, Union, List


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class RolloutExperience:
    """
    Represents a single rollout experience.

    state: list
    value: list
    action: list
    log_prob: list
    reward: list
    done: lis
    """
    state: List = field(default_factory=list)
    value: List = field(default_factory=list)
    action: List = field(default_factory=list)
    log_prob: List = field(default_factory=list)
    reward: List = field(default_factory=list)
    done: List = field(default_factory=list)

@dataclass
class RolloutMemory:
    """
    Represents a single rollout training samples.

    states: torch.Tensor
    values: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    """
    states: torch.Tensor
    values: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

class TrainingBuffer:
    """
    Represents a buffer of episode experiences.
    """
    def __init__(self) -> None:

        self.buffer = []
        self.step_counter = 0

    def add(self, episode_experience: Union[RolloutMemory, List[RolloutMemory]]) -> None:
        """
        adds the rollout of one or more episodes to the buffer
        Parameters
        ----------
        episode_experience : Union[RolloutMemory, List[RolloutMemory]]
            RolloutMemory or List of RolloutMemories each representing an episode

        Raises
        ------
        ValueError
            raised if episode_experience is neither RolloutMemory nor list of RolloutMemories

        """
        if isinstance(episode_experience, RolloutMemory):
            self.buffer.append(episode_experience)
            self.step_counter += len(episode_experience.advantages)
        elif isinstance(episode_experience, list):
            self.buffer.extend(episode_experience)
            self.step_counter += sum(len(thread.advantages) for thread in episode_experience)
        else:
            raise ValueError("Expected type of episode_experience to be either Experience or list of Experiences")

    def reset(self) -> None:
        """
        resets the buffer
        """
        self.step_counter = 0
        del self.buffer[:]

    def get_dataset(self) -> Batcher:
        """Creates the dataset for training the PPO agent

        Returns
        -------
        Batcher
            _description_
        """
        states = torch.cat([memory.states for memory in self.buffer], dim=0)
        actions = torch.cat([memory.actions for memory in self.buffer], dim=0)
        log_probs = torch.cat([memory.log_probs for memory in self.buffer], dim=0)
        advantages = torch.cat([memory.advantages for memory in self.buffer], dim=0)
        returns = torch.cat([memory.returns for memory in self.buffer], dim=0)

        return Batcher(states, actions, log_probs, advantages, returns)


    def __len__(self) -> int:
        """
        Returns the number of experience samples currently stored in the buffer

        Returns
        -------
        int
            length of the buffer
        """

        return self.step_counter

    def __getitem__(self, index: int) -> RolloutExperience:
        """
        Returns the experience sample at the given index

        Parameters
        ----------
        index : int
            index of the experience sample

        Returns
        -------
        RolloutExperience
            experience sample at the given index
        """
        return self.buffer[index]



class Batcher:
    """
    Represents a rollout dataset for training the PPO agent.

    """
    def __init__(self, states: torch.Tensor, actions: torch.Tensor, log_probs: torch.Tensor, advantages: torch.Tensor, returns: torch.Tensor) -> None:
        self.states = states
        self.actions = actions
        self.log_probs = log_probs
        self.advantages = advantages
        self.returns = returns

        self.batch_idx = np.random.permutation(len(self))
        self.batch_start = 0

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset

        Returns
        -------
        int
            number of samples in the dataset
        """

        return int(self.returns.nelement())

    def restart(self) -> None:
        """
        Clears the dataset
        """

        self.batch_idx = np.random.permutation(len(self))
        self.batch_start = 0

    def next_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the next batch of the dataset

        Parameters
        ----------
        batch_size : int
            size of the batch

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            batch of states, actions, log_probs, advantages and returns
        """

        buffer_len = len(self)
        batch_end = min(self.batch_start + batch_size, buffer_len)
        idx = self.batch_idx[self.batch_start:batch_end]
        self.batch_start = batch_end
        return (self.states[idx, :], self.actions[idx, :], self.log_probs[idx], self.advantages[idx], self.returns[idx])
