# Continuous Control with Reinforcement Learning

## Section 1: Getting Started
### Dependencies
```
torch==1.0.0
numpy==1.19.5
```

## Section 2: Download the Unity Environment
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

#### Version 1: One (1) Agent
- Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
#### Version 2: Twenty (20) Agents
- Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Then, place the file in the p2_continuous-control/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out this [link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

## Section 3: Implemented Algorithms
This repositeory provides the Training code, learned model and Training demonstration of the following algorithms:

### Proximal Policy Optimization
This used implementation of the PPO algorithm uses the following modules:
- `network.py` implements the `PPOPolicy` class comprised of separate Actor and Critic Networks. The exploratory standard deviation that alters the chosen actions during the training phase are made with trainable parameters. These trainable paramenters represent the logarithm of the noise std. This guarantees that the once exponentiated the std is strictly positive (it is unlikely that the parameters goes to -inf).
- `agent.py` implements `PPOAgent` class that encompasses the behaviour of the agent (performing an action on the environment, running a test episode on the environment).
- `trainer.py` Implements the `PPOTrainer` containing all the functionality for training an agent through the PPO algorithm. It builds a rollout of execution to then compute advantage and return. The rollout is used to build a dataset containing advantage, return, actions and log_prob of the steps taken. Those are then used to train the agent policy for multiple epochs.
- `buffer.py` Implements the `Batcher` module that generates the batches from the collected rollout dataset.

### Deep Deterministic Policy Gradient
This used implementation of the PPO algorithm uses the following modules:
- `network.py` implements `DDPGPolicy` network and its components `Actor` and `Critic`.
- `agent.py` implements `DDPGAgent` class that encompasses the behaviour of the agent (performing an action on the environment, running a test episode on the environment).
- `trainer.py` Implements the `DDPGTrainer` containing all the functionality for training an agent through the DDPG algorithm. It builds a training buffer of execution. The buffer is sampled to train the agent policy for multiple epochs. The trained policy belongs to a clone of the trained agent. The agent is improved by applying soft update from the cloned agent.
- `buffer_replay.py` implements `ReplayBuffer` and `PrioritizedReplayBuffer` who collect the data of coming from the training rollouts. The main difference is how the training samples are collected from the buffers. In the first case it's just a FIFO queue uniformly sampled. In the second a priority value is assigned on each sample. The samples are finally collected based on that priority value. The priority value is updated after each sample.
## Section 4: Results
### PPO
### DDPG



<img src="figures/DDPG_Training_progression.png" alt="DDPG Training progression" style="border: 5px solid  gray; background: white">


## Section 5: Possible improvements
### PPO
### DDPG

---
The README describes the the project environment details (i.e., the state and action spaces, and when the environment is considered solved).

The README has instructions for installing dependencies or downloading needed files.
