import torch
from torch import nn
import logging
import numpy as np
from collections import deque
import torch.optim as optim
import torch.nn.functional as F

from unityagents import UnityEnvironment
from agent import PPOAgent


LOGGER = logging.getLogger(__name__)

class TrainingBuffer():
    def __init__(self, device):
        
        self.__actions = []
        self.__states = []
        self.__action_log_probs = []
        self.__rewards = []
        self.__state_values = []
        self.__dones = []
        self.device = device

    def append(self, action, state, log_prob, reward, value, done):
        
        self.__state_values.append(value)
        self.__rewards.append(reward)
        self.__dones.append(done)
        self.__action_log_probs.append(log_prob)
        self.__states.append(state)
        self.__actions.append(action)

    def clear(self):
        del self.__actions[:]
        del self.__states[:]
        del self.__action_log_probs[:]
        del self.__state_values[:]
        del self.__rewards[:]
        del self.__dones[:]

    def __len__(self):
        return len(self.__rewards)
    
    def get_batch(self):
        return (self.actions, self.states, self.log_probs, self.state_values, self.rewards, self.dones)
    
    @property
    def actions(self):
        return torch.cat(self.__actions, dim=0).to(self.device)
    
    @property
    def states(self):
        return torch.cat(self.__states, dim=0).to(self.device)
    
    @property
    def log_probs(self):
        return torch.cat(self.__action_log_probs, dim=0).to(self.device).detach()
    
    @property
    def state_values(self):
        return torch.cat(self.__state_values).to(self.device).detach()
    
    @property
    def rewards(self):
        return torch.cat(self.__rewards).to(self.device)
    
    @property
    def dones(self):
        return torch.cat(self.__dones).to(self.device)


class PPOTrainer():
    def __init__(self, agent: PPOAgent, learning_rate: float=1e-3, training_epochs: int=10, 
                 gamma: float=0.99, epsilon_clip: float=.1, beta: float = 1e-3, std_decay_rate=(1-5e-3),
                 use_normalized_advantage: bool=True, use_generalized_advantage: bool=True, gae_lambda=0.7):
        
        self.agent = agent
        self.optimizer = optim.Adam([
            {'params': self.agent.policy.stub.parameters(), 'lr': learning_rate},
            {'params': self.agent.policy.actor_head.parameters(), 'lr':learning_rate},
            {'params': self.agent.policy.critic_head.parameters(), 'lr':learning_rate}
            ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epsilon_clip = epsilon_clip
        self.gamma = gamma
        self.training_epochs = training_epochs
        self.use_normalized_advantage = use_normalized_advantage
        self.use_generalized_advantage = use_generalized_advantage
        self.gae_lambda = gae_lambda if use_generalized_advantage else None
        self.beta = beta
        self.buffer = TrainingBuffer(self.device)
        self.__debug = False

        self.std_decay_rate = std_decay_rate

        self.agent.policy.to(self.device)
        self.__brain_name = self.agent.env.brain_names[0]
        self.__brain = self.agent.env.brains[self.__brain_name]

        LOGGER.info('\t' + '-' * 31 + ' Params ' + '-' * 31)
        LOGGER.info(f'\tEpsilon clip: {self.epsilon_clip}\t Gamma: {self.gamma}\t Learning Rate: {learning_rate}')
        LOGGER.info(f'\tTraining epochs: {self.training_epochs} ')
        LOGGER.info(f'\tUsing Normalized Advantage: {self.use_normalized_advantage} ')
        LOGGER.info(f'\tUsing General Advantage Estimation: {self.use_generalized_advantage} ')
        LOGGER.info('\t' + '-' * 70)
    

    def toggle_debug(self):
        self.__debug = not self.__debug
        state_string = 'ON' if self.__debug else 'OFF'
        LOGGER.setLevel(logging.DEBUG if self.__debug else logging.INFO)
        LOGGER.debug(f'Debug mode is : {state_string}')

    def update(self):

        old_states = self.buffer.states
        old_actions = self.buffer.actions
        old_log_probs = self.buffer.log_probs
        old_state_values = self.buffer.state_values

        if self.use_generalized_advantage:
            returns, advantages = self.compute_general_advantage()
        else:
            returns = self.compute_MCreturns()
            
            returns = self.normalize(returns)
            advantages = returns.detach() - old_state_values

        if self.use_normalized_advantage:
            # advantages = self.normalize(advantages)
            returns = self.normalize(returns)


        for epoch in range(self.training_epochs):
            
            new_log_probs, new_state_values, dist_entropy = self.agent.policy.evaluate(old_states, old_actions)

            ratio = torch.exp(new_log_probs - old_log_probs.detach())

            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2)

            critic_loss = F.mse_loss(new_state_values.squeeze(), returns.squeeze())

            loss = actor_loss + 0.5 * critic_loss - self.beta * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            self.buffer.clear()
            if self.__debug:
                # LOGGER.debug(f'surrogate1: {surrogate1}, surrogate_clipped: {surrogate2}')
                LOGGER.debug(f'\tEpoch: {epoch}, actor_loss: {actor_loss.mean():.6f}, critic_loss: {critic_loss.mean():.6f}, entropy_bonus: {dist_entropy.mean():.6f}')

    
    def compute_MCreturns(self) -> torch.Tensor:
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        returns = torch.zeros_like(rewards)
        R = 0
        for step in (reversed(range(len(rewards)))):
            R = rewards[step] + (self.gamma * R * (1 - dones[step]))
            returns[step] = R
        return returns
    
    def compute_general_advantage(self):
        assert(self.gae_lambda is not None, "If compute_general_advantage is set to True a gae_lambda value should be provided")
        rewards = self.buffer.rewards
        values = self.buffer.state_values
        advantages = torch.zeros_like(rewards).to(self.device)
        gae = next_value = 0.
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value - values[step]
            advantages[step] = gae = delta + self.gamma * self.gae_lambda * gae
            next_value = values[step]

        returns = advantages + values
        return returns, advantages


    def normalize(self, batch:torch.Tensor, non_zero_eps: float=1e-8):

        mean = torch.mean(batch)
        stdev = torch.std(batch) + non_zero_eps

        return (batch - mean) / stdev
    
    
    def train(self, n_episodes: int=1000, max_t: int=100, update_every: int=200, print_every: int=100, target:float = 30.):
        
        LOGGER.info('\t' + '-' * 30 + ' Training ' + '-' * 30)
        LOGGER.info(f'\tN Episodes {n_episodes} , max length episode: {max_t}')
        LOGGER.info(f'\tUpdate every {update_every} steps')
        LOGGER.info(f'\tPrint every {print_every} episodes')
        LOGGER.info(f'\tTarget score: {target}')
        LOGGER.info('\t' + '-' * 70)
        scores_deque = deque(maxlen=100)
        scores = []

        for episode in range(1, n_episodes + 1):
            try:
                rewards = self.run_training_episode(max_t, update_every)

                scores.append(sum(rewards))
                scores_deque.append(sum(rewards))

                if episode % print_every == 0:
                    LOGGER.info(f'\t ---> Episode\t {episode}\tAverage Score: {np.array(scores_deque)[-print_every:].mean():.2f}')
                if np.mean(scores_deque) >= target:
                    LOGGER.info(f'\tEnvironment solved in {episode - 100:d} episodes!\tAverage Score: {np.mean(scores_deque):.2f}')
                    break
            except KeyboardInterrupt:
                LOGGER.info(f"\tTraining Interrupted at episode {episode}")
                break
        return scores

    def run_training_episode(self, max_t: int, update_every: int) -> np.array:
        env_info = self.agent.env.reset(train_mode=True)[self.__brain_name] # reset the environment
        state = torch.from_numpy(env_info.vector_observations).float()
        done = False
        
        rewards = []
        for t in range(max_t):
            action, log_prob = self.agent.get_action(state)
            next_state, reward, done = self.agent.act(action)
            state_value = self.agent.get_value(state)
            
            rewards.append(np.asarray(reward))
            self.buffer.append(
                action, state, log_prob, 
                reward, state_value, done)

            state = next_state
            if len(self.buffer) >= update_every:
                LOGGER.debug(f'\tUpdating the nework ...')
                self.update()
                self.agent.decay_action_std(self.std_decay_rate)
            if done:
                break

        return np.concatenate(rewards)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15,5))
    env = UnityEnvironment(file_name='/Users/claudiocoppola/code/RL_repos/Reacher.app', no_graphics=True)
    agent = PPOAgent(env, network_config={'hidden_sizes':[256, 256], 'hidden_activation': nn.Tanh})

    trainer = PPOTrainer(agent, gamma=0.3, epsilon_clip=0.2, training_epochs=30)
    scores = trainer.train(n_episodes=10000, max_t=200)
    ax.plot(scores)

    plt.show()
    # means, stds, values, states, actions, rewards, masks = agent.run_episode(100)
    # print(sum(rewards))