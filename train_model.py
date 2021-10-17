from unityagents import UnityEnvironment
from src.agents import Agent, ReplayDDQNAgent, PriorityReplayDDQNAgent
from src.environment_utils import Execution_Manager
from random import randint
from src import package_path


LAYERS = [64, 128, 64]
ENV_PATH = package_path + "/unity/Crawler.app"

out_file = 'trained_model.pth'


checkpoint_path = package_path + '/assets/models/{}'.format(out_file)
plot_fig_path = package_path + '/assets/figs/{}'.format(out_file.split('.')[0] + '.svg')