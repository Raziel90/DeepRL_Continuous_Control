import logging
from tqdm import trange
from torch import optim
import torch
import numpy as np
from collections import namedtuple
from typing import List

from .agent import PPOAgent
from .buffer import TrainingBuffer, RolloutExperience, RolloutMemory
from collections import deque


CRITIC_WEIGHT_DECAY = 2e-6
BATCH_SIZE = 512

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOGGER = logging.getLogger(__name__)

class PPOTrainer:
    def __init__(self, agent: PPOAgent, learning_rate_actor: float = 1e-3, learning_rate_critic: float = 1e-3, epsilon_clip:float=0.1, beta: float=1e-3,
                 rollout_length: int = 1000, batch_size=1024, gamma: float = 0.99, gae_lambda: float = 1.0, training_epochs: int = 10,
                 filename='PPO.pth', filepath='./') -> None:
        """
        PPO Trainer class.


        Parameters
        ----------
        agent : PPOAgent
            Agent to train with PPO
        learning_rate_actor : float, optional
            learning rate used to optimize the actor network, by default 1e-3
        learning_rate_critic used to optimize the critic network: float, optional
            learning rate , by default 1e-3
        epsilon_clip : float, optional
            PPO epsilon paramenter, by default 0.1
        beta : float, optional
            entropy bonus coefficient, by default 1e-3
        rollout_length : int, optional
            minimum amount of samples that should in a rollout, by default 1000
        batch_size : int, optional
            length of the minibatches, by default 1024
        gamma : float, optional
            discount factor, by default 0.99
        gae_lambda : float, optional
            lambda of the generalized advantage estimation, by default 1.0
        training_epochs : int, optional
            number of epochs to train with at every update, by default 10
        filename : str, optional
            filename of the network checkpoint, by default 'PPO.pth'
        filepath : str, optional
            path of the checkpoint, by default './'
        """

        self.agent = agent

        self.gamma = gamma
        self.beta = beta
        self.n_epochs = training_epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        self.epsilon_clip = epsilon_clip
        self.filename = filename
        self.filepath = filepath
        self.gae_lambda = gae_lambda
        self.buffer = TrainingBuffer()

        self.scores = []
        self.score_queue = deque(maxlen=100)

        self.optimizer = optim.Adam([
            {"params": self.agent.policy.pi.parameters(), "lr": learning_rate_actor, "weight_decay": CRITIC_WEIGHT_DECAY},
            {"params": self.agent.policy.V.parameters(), "lr": learning_rate_critic, "weight_decay": CRITIC_WEIGHT_DECAY},
            {"params": self.agent.policy.log_std, "lr": 1e-3}])

        self.__brain_name = self.agent.env.brain_names[0]
        self.__brain = self.agent.env.brains[self.__brain_name]

    def train(self, n_episodes: int, max_steps_per_episode: int, target: float = 30.) -> PPOAgent:
        """begins the training of the agent.
        Terminates when the 100 episode mean of at least one agent reaches the target or
        when interruption signal is received.

        Parameters
        ----------
        n_episodes : int
            The maximum number of episodes to train on
        max_steps_per_episode : int
            max number of steps to perform within an episode
        target : float, optional
            target score to stop training at, by default 30.

        Returns
        -------
        PPOAgent
            Trained Agent
        """
        pbar = trange(n_episodes, desc='Training', unit='episodes', leave=True)
        pbar.set_description(
                f"Episode {0}: Exploration std mean: {0.:.2f}, "
                f"Score: {0.:.2f}, "
                f"100 episode Mean Score: {0.:.2f}")
        for episode in pbar:
            try:
                score = self.run_training_episode(max_steps_per_episode=max_steps_per_episode)
                self.scores.append(score)
                self.score_queue.append(score)
                if len(self.buffer) >= self.rollout_length:
                    self.update_policy()
                    self.buffer.reset()

            except KeyboardInterrupt:
                pbar.close()
                LOGGER.info(f"\tTraining Interrupted at episode {len(self.scores)}")
                break

            finally:
                self.agent.to_file((self.filepath + '/' + self.filename))
                if np.all(np.mean(self.score_queue, axis=0) >= target) and len(self.scores) > 100:
                    LOGGER.info(f"\tTraining Complete at episode {len(self.scores)}")
                    break

            pbar.set_description(
                f"Episode {episode + 1}: Exploration std mean: {(self.agent.policy.log_std.exp().data.mean()):.2f}, "
                f"Score: {score.mean():.2f}, "
                f"100 episode Mean Score: {np.mean(self.score_queue, axis=0).mean():.2f}")

        return self.agent

    def run_training_episode(self, max_steps_per_episode: int) -> np.ndarray:
        """Runs one training episode

        Parameters
        ----------
        max_steps_per_episode : int
            maximum number of steps to perform in the episode

        Returns
        -------
        np.ndarray
            (1 X n_thread) array of scores for the episode

        """
        env_info = self.agent.env.reset(train_mode=True)[self.__brain_name] # reset the environment
        state = np.asarray(env_info.vector_observations)

        done = False
        rollout = [RolloutExperience() for _ in range(self.agent.n_threads)]
        rewards = []
        for _ in range(max_steps_per_episode):
            action, log_prob, next_state, reward, done = self.agent.act(state, noisy=True)
            value = self.agent.policy.state_value(state).detach().numpy()
            for thread in range(self.agent.n_threads):
                rollout[thread].state.append(state[thread, :][None, :])
                rollout[thread].value.append(value[thread])
                rollout[thread].action.append(action[thread, :][None, :])
                rollout[thread].log_prob.append(log_prob[thread].unsqueeze(0))
                rollout[thread].reward.append(reward[thread])
                rollout[thread].done.append(done[thread])
            state = next_state
            rewards.append(reward.squeeze())
            if np.array(done).astype(bool).any():
                break
        incumbent_value = self.agent.policy.state_value(state).detach().numpy()

        for thread, roll in enumerate(rollout):
            roll.reward = np.array(roll.reward)
            roll.reward = (roll.reward - np.mean(roll.reward)) / (np.std(roll.reward) + 1e-6)
            advantage = torch.from_numpy(self.compute_general_advantage(roll, incumbent_value[thread], True)).float()
            values = torch.from_numpy(np.concatenate(roll.value, axis=0)).float()
            memory = RolloutMemory(
                states=torch.from_numpy(np.concatenate(roll.state, axis=0)).float(),
                actions=torch.cat(roll.action, dim=0),
                log_probs=torch.cat(roll.log_prob, dim=0).float(),
                advantages=advantage,
                returns=advantage + values
            )
            self.buffer.add(memory)

        return np.array(rewards).sum(axis=0)

    def compute_general_advantage(self, rollout: RolloutExperience, last_value: float=0.0, normalize: bool=True) -> np.ndarray:
        """computes GAE starting from the rollout and the estimate of the value
        of the final state. if normalize is True, the advantages are normalized
        by the standard deviation of the advantages used in the update_policy function.
        The advantages are computed in reverse order as follows:
        advantages[t] = rewards[t] + gamma * value[t + 1] * (1 - done[t]) - value[t]
        where value[t + 1] is the estimate of the value of the state at time t + 1.

        Parameters
        ----------
        rollout :
            list of RolloutExperience
        last_value : float, optional
            the value of the last state reached during the rollout, by default 0.0
        normalize : bool, optional
            if true the advantage is normalized, by default True

        Returns
        -------
        np.ndarray
            (1 X n_thread) array of advantages for the rollout

        Examples
        --------
        >>> rollout = RolloutExperience()
        >>> rollout.reward = [1, 2, 3]
        >>> rollout.value = [1, 2, 3]
        >>> rollout.done = [False, False, True]
        >>> compute_general_advantage(rollout, normalize=False)
        array([ 0.,  1.,  2.])
        >>> compute_general_advantage(rollout, normalize=True)
        array([ 0.        ,  0.66666667,  1.33333333])

        """
        advantages = np.zeros((len(rollout.reward)))

        gae = next_value = last_value
        for step in reversed(range(len(rollout.reward))):
            delta = rollout.reward[step] + self.gamma * next_value * (1 - rollout.done[step]) - rollout.value[step]
            advantages[step] = gae = delta + self.gamma * self.gae_lambda * gae
            next_value = rollout.value[step]

        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)#max(advantages.std(), 1e-4)
        return advantages

    def update_policy(self) -> None:
        """
        Updates the policy using the buffer.
        The update is done for n_epochs number of epochs.
        The loss is computed as follows:
        loss = actor_loss + 0.5 * critic_loss - beta * entropy
        where actor_loss is the negative of the mean of the surrogate of the ratio
        between the new log probability and the old log probability.
        critic_loss is the negative of the mean of the difference between the
        new state value and the old state value.
        The entropy is the negative of the mean of the entropy of the policy.
        The update is done using the Adam optimizer.
        The loss is also printed in the progress bar.

        Examples
        --------
        >>> trainer = Trainer(agent, buffer)
        >>> trainer.update_policy()

        Notes
        -----
        The loss is computed using the Adam optimizer.
        The loss is also printed in the progress bar.

        References
        ----------
        [1] https://arxiv.org/abs/1707.06347

        Returns
        -------
        None.


        """

        batcher = self.buffer.get_dataset()
        update_tracker = trange(self.n_epochs, desc='Updating Policy', leave=False, unit='epoch')
        for _ in update_tracker:
            batcher.restart()
            cum_loss_total = 0.0
            cum_loss_actor = 0.0
            cum_loss_critic = 0.0
            while batcher.batch_start < len(batcher):

                (old_states, old_actions, old_log_probs, old_advantages, old_returns) = (
                    batcher.next_batch(self.batch_size))

                new_log_probs, entropy = self.agent.policy.action_prob(old_states, old_actions)
                new_state_values = self.agent.policy.state_value(old_states)
                ratio = (new_log_probs - old_log_probs.detach()).exp()
                surrogate1 = ratio * old_advantages.detach()
                surrogate2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * old_advantages.detach()
                actor_loss = - torch.min(surrogate1, surrogate2).mean()
                critic_loss = (new_state_values.squeeze() - old_returns.squeeze()).pow(2).mean()

                loss = actor_loss + 0.5 * critic_loss - self.beta * entropy.mean()
                cum_loss_total += loss.mean().item()
                cum_loss_actor += actor_loss.mean().item()
                cum_loss_critic += critic_loss.mean().item()


                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

            update_tracker.set_description(
                "Loss - "
                f"Total: {cum_loss_total:.2f}"
                f" Critic: {cum_loss_critic:.2f}"
                f" Actor: {cum_loss_actor:.2f}"
                )

if __name__ == "__main__":
    env = UnityEnvironment(file_name='/Users/claudcop/code/deep-rl/DeepRL_Continuous_Control/unity/Reacher20.app', no_graphics=True)

    agent = PPOAgent(env, seed=4)
    agent.run_test_episode(max_steps=1000, track_progress=True)
    trainer = PPOTrainer(agent, rollout_length=1000)
    print('END!')
