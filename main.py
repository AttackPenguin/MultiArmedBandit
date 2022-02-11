import os

import numpy as np
import torch
from torch import nn


def reward_f_factory(n: int,
                     std: float=0.2):
    distributions = list()
    for _ in range(n):
        dist = np.random.normal(
            loc=np.random.rand(),
            scale=std
        )
        distributions.append(dist)

    def get_reward(n: int):

        f_dists = distributions
        return_val = f_dists[n]
        if return_val < 0:
            return 0.0
        if return_val > 1:
            return 1.0

    return get_reward


class MultiArmedBandit(nn.Module):
    def __init__(self,
                 n: int,
                 dist_method: callable):
        super(MultiArmedBandit, self).__init__()
        # We will store a list of modules that will pull n-1 levers. Our
        # first lever pull will be random, so there is no reason to
        # implement a module for it.
        self.n = n
        self.modules = nn.ModuleList()
        for i in range(1, n):
            module = nn.Sequential(
                nn.Linear(i * (n + 1), 256),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(256, n),
                nn.Softmax(n)
            )
            self.modules.append(module)

    def forward(self,
                dist_methods: list[callable]):
        levers = list()
        rewards = list()

        module_input = torch.rand((len(dist_methods), self.n))
        # add the lever pull output to each sub-tensor
        for i, x in enumerate(module_input):
            lever = (x == torch.max(x)).nonzero(as_tuple=True)[0]
            levers.append([x])
            reward = dist_methods[i](lever)
            rewards.append([reward])
            module_input[i] = torch.cat((x, torch.Tensor(reward)))
        module_output = None

        for module in self.modules:
            if module_output is not None:
                torch.cat(module_output)
            module_output = module(input)
            # add the lever pull output to each sub-tensor
            for i, x in enumerate(module_output):
                lever = (x == torch.max(x)).nonzero(as_tuple=True)[0]
                levers[i].append(x)
                reward = dist_methods[i](lever)
                rewards[i].append(reward)
                module_output[i] = torch.cat((x, torch.Tensor(reward)))

        return levers, rewards
