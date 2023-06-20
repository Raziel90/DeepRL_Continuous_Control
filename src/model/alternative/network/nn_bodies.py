import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_components import NoisyLinear


class BodyBase(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x
    
    def init_weights(self, w_scale=1.):
        pass

class FC_Body(BodyBase):
    def __init__(self, state_dim, hidden_units=[64, 64], gate=F.relu, noisy_linear=False):
        super().__init__()
        dims = [state_dim] + hidden_units

        if noisy_linear:
            self.layers = nn.ModuleList(
                [nn.NoisyLinear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        else:
            self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]
        self.noisy_linear = noisy_linear

        self.init_weights()

    def reset_noise(self):
        if self.noisy_linear:
            for layer in self.layers:
                layer.reset_noise()
    
    def forward(self, x):
        
        y = [self.gate(layer(x)) for layer in self.layers]
        
        return y[-1]
    
    def init_weights(self, w_scale=1.):
        for layer in self.layers:
            if isinstance(layer, NoisyLinear):
                layer.reset_parameters()
            else:
                nn.init.orthogonal_(layer, layer.weight.data)
                layer.weight.data.mul_(w_scale)
                nn.init.constant_(layer.bias.data, 0)
    


