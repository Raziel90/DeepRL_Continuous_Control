
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

import torch
from unityagents import UnityEnvironment


class Execution_Manager():
    def __init__(self, agent, unity_env="/data/Banana_Linux_NoVis/Banana.x86_64"):
        if isinstance(unity_env, UnityEnvironment):
            self.env = unity_env
        elif isinstance(unity_env, str):
            self.env = UnityEnvironment(file_name=unity_env, seed=int(np.random.randint(1e6)))
        else:
            raise ValueError('unity_env must be a string path to the Unity environment or a UnityEnvironment instance.')
        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.agent = agent
        self.train_scores = []