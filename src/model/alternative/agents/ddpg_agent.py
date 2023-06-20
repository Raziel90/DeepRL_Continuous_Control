from .base_agent import BaseAgent



class DDPGAgent(BaseAgent):
    def __init__(self, network_factory, replay_buffer_class, eval_episodes, 
                 normalizer_class, interpolation_factor, env):
        super().__init__(eval_episodes, normalizer_class, env)

        # self.task = config.task_fn()
        self.network = network_factory()
        self.target_network = network_factory()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = replay_buffer_class()
        self.interpolation_factor = interpolation_factor
        # self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None

    def soft_update(self):
        for target_param, param in zip(self.target_network(), self.network.parameters()):
            target_param.detach_()
            target_param.copy_(
                target_param * (1. - self.interpolation_factor) +
                param * (self.interpolation_factor))
            
    def eval_step(self, state):
        self.state_normalizer
