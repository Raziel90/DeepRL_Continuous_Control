from unityagents import UnityEnvironment
from src.agent import Agent, DDPGAgent
from src.environment import Execution_Manager
from random import randint
from src import package_path


LAYERS = [64, 128, 64]
ENV_PATH = package_path + "/unity/Crawler.app"

out_file = 'trained_model.pth'


checkpoint_path = package_path + '/assets/models/{}'.format(out_file)
plot_fig_path = package_path + '/assets/figs/{}'.format(out_file.split('.')[0] + '.svg')


if __name__ == '__main__':

    unity_env = UnityEnvironment(ENV_PATH, seed=randint(0, 1e6))
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]
    env_info = unity_env.reset(train_mode=True)[brain_name]
    states_dim = len(env_info.vector_observations[0])
    action_dim = brain.vector_action_space_size

    agent = DDPGAgent(state_size=states_dim, action_size=action_dim, hidden_size=LAYERS, seed=0)

