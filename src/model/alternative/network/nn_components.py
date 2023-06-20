import torch
import torch.nn as nn
import torch.nn.functional as F

import math

NOISY_LAYER_STD = 0.1


class NoisyLinear(nn.Module):
    def __init__(self, in_feats, out_feats, std_init=0.3):
        super().__init__()

        self.in_dim = in_feats
        self.out_dim = out_feats
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.zeros((out_feats, in_feats)), requires_grad=True)
        self.weight_sigma = nn.Parameter(torch.zeros((out_feats, in_feats)), requires_grad=True)
        self.register_buffer('weight_epsilon', torch.zeros((out_feats, in_feats)))

        self.bias_mu = nn.Parameter(torch.zeros(out_feats), requires_grad=True)
        self.bias_sigma = nn.Parameter(torch.zeros(out_feats), requires_grad=True)
        self.register_buffer('bias_epsilon', torch.zeros(out_feats))
        
        self.register_buffer('noise_in', torch.zeros(in_feats))
        self.register_buffer('noise_out_weight', torch.zeros(out_feats))
        self.register_buffer('noise_out_bias', torch.zeros(out_feats))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu 
            bias = self.bias_mu

        return F.linear(x, weight=weight, bias=bias)
    
    def reset_parameters(self):
        mu_range = 1/ math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(1)))

    def reset_noise(self):
        self.noise_in.normal_(std=NOISY_LAYER_STD)
        self.noise_out_weight.normal_(std=NOISY_LAYER_STD)
        self.noise_out_bias.normal_(std=NOISY_LAYER_STD)

        self.weight_epsilon.copy_(
            self.transform_noise(self.noise_out_weight).ger(self.transform_noise(self.noise_in))
            )
        
        self.bias_epsilon.copy_(
            self.transform_noise(self.noise_out_bias)
            )
        
    def transform_noise(self, x):
        return x.sign().mul(x.abs().sqrt())
        


