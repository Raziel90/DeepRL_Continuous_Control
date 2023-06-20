from unityagents import UnityEnvironment
import logging
import numpy as np
# from src.agents import Agent, ReplayDDQNAgent, PriorityReplayDDQNAgent
# from src.environment import Execution_Manager

from src import package_path


LAYERS = [64, 64]
ENV_PATH = package_path + "/unity/Reacher.app"

# package_path = '/'.join(__file__.split('/')[:-1])

in_file = 'trained_model.pth'
checkpoint_path = package_path + '/assets/models/{}'.format(in_file)


def get_env_metadata(unity_env_info):
    unity_env_info
    num_agents = len(unity_env_info.agents)
    logging.info(f'Number of agents:{num_agents}')

    # size of each action
    action_size = brain.vector_action_space_size
    logging.info(f'Size of each action:{action_size}')

    # examine the state space 
    states = unity_env_info.vector_observations
    state_size = states.shape[1]
    logging.info('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    return action_size, state_size
    

if __name__ == '__main__':
    print(package_path)
    env = UnityEnvironment(file_name=ENV_PATH)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    action_size, state_size = get_env_metadata(env_info)


    
    print(action_size, state_size)