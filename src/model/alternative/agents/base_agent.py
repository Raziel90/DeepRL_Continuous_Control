import torch
import pickle
import logging
import numpy as np
# from torch.utils.tensorboard import SummaryWriter


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()

class BaseAgent:
    def __init__(self, eval_episodes, normalizer_class, env):
        # self.config = config
        self.logger = LOGGER
        self.evaluation_episodes = eval_episodes
        self.state_normalizer = normalizer_class()
        self.env  = env
        self.task_ind = 0

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.state_normalizer.state_dict(), f)

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state):
        raise NotImplementedError()
    
    def eval_episode(self):
        state = self.env.reset()
        done = False
        while not done:
            action = self.eval_step(state)
            state, reward, done, info = self.env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        return ret
    
    def eval_episodes(self):
        episodic_returns = []
        for ep in range(self.evaluation_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
            self.logger.info(
                f'steps {self.total_steps}, episodic_return {np.mean(episodic_returns)}({np.std(episodic_returns) / np.sqrt(len(episodic_returns))})')
            return {'episodic_return_test': np.mean(episodic_returns)}

    