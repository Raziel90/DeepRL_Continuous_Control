from unityagents import UnityEnvironment
from src.agents import Agent, ReplayDDQNAgent, PriorityReplayDDQNAgent
from src.environment import Execution_Manager

from src import package_path


LAYERS = [64, 64]
ENV_PATH = package_path + "/unity/Banana.app"

# package_path = '/'.join(__file__.split('/')[:-1])

in_file = 'trained_model.pth'
checkpoint_path = package_path + '/assets/models/{}'.format(in_file)


if __name__ == '__main__':
    print(package_path)