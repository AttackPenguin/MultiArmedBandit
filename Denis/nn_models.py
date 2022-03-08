from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from Denis.reward_generators import RewardGenerator


class MABInceptionModel(nn.Module):
    def __init__(self,
                 n: int = 10,
                 pulls: int = 100,
                 module_width: int = 256,
                 use_dropout: bool = False,
                 dropout_ratio: float = 0.5):
        super(MABInceptionModel, self).__init__()
        # We will store a list of modules that will pull (pulls-1) levers. Our
        # first lever pull will be random, so there is no reason to
        # implement a module for it.
        self.n = n
        self.pulls = pulls
        self.module_width = module_width
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio
        self.modules_list = nn.ModuleList()
        for i in range(1, pulls):
            components = OrderedDict()

            # Each module takes the output of all prior modules,
            # plus their calculated rewards, so we increase the expected
            # input size as we add modules.
            components[f'mod{i}lin1'] = nn.Linear(i * (n + 1), module_width)
            components[f'mod{i}rrelu1'] = nn.RReLU()
            if use_dropout:
                components[f'mod{i}dropout1'] = nn.Dropout(p=dropout_ratio)

            components[f'mod{i}lin2'] = nn.Linear(module_width, module_width)
            components[f'mod{i}rrelu2'] = nn.RReLU()
            if use_dropout:
                components[f'mod{i}dropout2'] = nn.Dropout(p=dropout_ratio)

            components[f'mod{i}lin3'] = nn.Linear(module_width, module_width)
            components[f'mod{i}rrelu3'] = nn.RReLU()
            if use_dropout:
                components[f'mod{i}dropout3'] = nn.Dropout(p=dropout_ratio)

            # The final linear layer drops down to n output values,
            # to be used to select a lever to pull.
            components[f'mod{i}lin4'] = nn.Linear(module_width, n)
            components[f'mod{i}rrelu4'] = nn.RReLU()
            if use_dropout:
                components[f'mod{i}dropout4'] = nn.Dropout(p=dropout_ratio)

            components[f'mod{i}softmax'] = nn.Softmax(dim=2)

            module = nn.Sequential(components)
            self.modules_list.append(module)

    def forward(self,
                reward_generators: list[RewardGenerator]):
        """
        The forward method gets a list of RewardGenerator objects, rather than
        a tensor of input values. The reward generators calculate a reward
        based on the lever pulled by each module in self.modules_list.
        Input tensors are constructed from random floats in the range [0, 1],
        resulting in a random first lever pull.
        :param reward_generators: A list of RewardGenerator methods.
        :return:
        """
        # We will store rewards and lever choices in lists and return them as
        # well, to facilitate analysis of data. Each row is specific to a
        # RewardGenerator in reward_generators.
        rewards = list()
        levers = list()

        # We initialize an input tensor to random values in the range [0, 1]
        # len(reward_generators) is our batch size.
        module_input = torch.rand(
            (len(reward_generators), self.n), requires_grad=True
        )

        # We iterate through each of the input tensors, calculating which
        # lever is pulled and the resulting reward. We append these results
        # to all_rewards and levers. This is our initialization step to
        # create our input for our first module.
        for i, x in enumerate(module_input):
            max_indices = list((x == torch.max(x)).nonzero(as_tuple=True)[0])
            # There is a very small chance of getting two identical values in
            # our input vector, so we invoke np.random.choice to deal with
            # this possibility.
            lever = np.random.choice(max_indices)
            levers.append([lever])
            reward = reward_generators[i].get_reward(lever)
            rewards.append([reward])

        # Our model expects inputs in the shape: (batch size, 1, num levers + 1)
        # We need "num levers + 1" because we also feed the reward value from
        # the previous layer's output into each module.
        module_input = module_input.reshape(
            [len(reward_generators), 1, self.n]
        )
        # We can use our rewards list here. When we iterate through the
        # modules below, we will need to locally track them in a
        # local_rewards list because rewards will accumulate multiple columns
        # of outputs.
        local_rewards = torch.Tensor(rewards).reshape(
            [len(reward_generators), 1, 1]
        )
        module_input = torch.cat(
            [module_input, local_rewards], dim=2
        )

        # module_outputs will accumulate the output layers of the modules. We
        # will return this Tensor, and our loss and optimization methods will
        # use it to modify parameter weights. Each row of this tensor will be
        # specific to a RewardGenerator in reward_generators.
        module_outputs = None
        # We initialize module_output to be None to facilitate identification
        # of our first pass.
        module_output = None

        # We iterate through each module in our list of modules, stacking the
        # output of each module onto it's input to create the input for the
        # next module.
        for pull, module in enumerate(self.modules_list):
            # After the first loop, we will want to concatenate our output to
            # our previous module's input to generate our new input for the
            # next module.
            if module_output is not None:  # First pass only
                module_input = torch.cat((module_input, module_output), dim=2)
            # Do a forward pass through the module.
            module_output = module(module_input)

            # For each RewardGenerator in the batch, determine the lever
            # pulled and the reward generated for pulling the lever.
            local_rewards = list()
            for i, x in enumerate(module_output):
                max_indices = list(torch.ravel(
                    (x == torch.max(x)).nonzero(as_tuple=True)[1]
                ))
                # There is a very small chance of getting two identical values
                # in our output vector, so we invoke np.random.choice to deal
                # with this possibility.
                lever = np.random.choice(max_indices)
                levers[i].append(lever)
                reward = reward_generators[i].get_reward(lever)
                rewards[i].append(reward)
                local_rewards.append([reward])

            # We append module_output to module_outputs BEFORE tacking the
            # reward values on, since we won't be back-propagating through
            # the rewards, just the module outputs.
            if module_outputs is None:  # First pass only
                module_outputs = module_output
            else:
                module_outputs = torch.cat(
                    [module_outputs, module_output], dim=2
                )
            local_rewards = torch.Tensor(local_rewards).reshape(
                [len(reward_generators), 1, 1]
            )
            module_output = torch.cat(
                [module_output, local_rewards], dim=2
            )

        return module_outputs, levers, rewards


class MABInceptionModel2(nn.Module):
    """
    The major change in this model is that reward values are amplified as
    input by feeding them into a single dense layer before densely connecting
    to the block that will calculate the next lever pull.
    """

    def __init__(self,
                 n: int = 10,
                 pulls: int = 100,
                 module_width: int = 256,
                 use_dropout: bool = False,
                 dropout_ratio: float = 0.5,
                 use_batch_norm = False
    ):
        super(MABInceptionModel2, self).__init__()
        # We will store a list of modules that will pull (pulls-1) levers. Our
        # first lever pull will be random, so there is no reason to
        # implement a module for it.
        self.n = n
        self.pulls = pulls
        self.module_width = module_width
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio
        self.use_batch_norm = use_batch_norm
        self.modules_list = nn.ModuleList()
        for i in range(1, pulls):
            components = OrderedDict()

            # Each module takes the output of all prior modules,
            # plus their calculated rewards, so we increase the expected
            # input size as we add modules.
            components[f'mod{i}lin1'] = nn.Linear(i * (n + 1), module_width)
            if use_batch_norm:
                components[f'mod{i}bn1'] = nn.BatchNorm1d(module_width)
            components[f'mod{i}rrelu1'] = nn.RReLU()
            if use_dropout:
                components[f'mod{i}dropout1'] = nn.Dropout(p=dropout_ratio)

            components[f'mod{i}lin2'] = nn.Linear(module_width, module_width)
            if use_batch_norm:
                components[f'mod{i}bn2'] = nn.BatchNorm1d(module_width)
            components[f'mod{i}rrelu2'] = nn.RReLU()
            if use_dropout:
                components[f'mod{i}dropout2'] = nn.Dropout(p=dropout_ratio)

            components[f'mod{i}lin3'] = nn.Linear(module_width, module_width)
            if use_batch_norm:
                components[f'mod{i}bn3'] = nn.BatchNorm1d(module_width)
            components[f'mod{i}rrelu3'] = nn.RReLU()
            if use_dropout:
                components[f'mod{i}dropout3'] = nn.Dropout(p=dropout_ratio)

            # The final linear layer drops down to n output values,
            # to be used to select a lever to pull.
            components[f'mod{i}lin4'] = nn.Linear(module_width, n)
            components[f'mod{i}rrelu4'] = nn.RReLU()

            components[f'mod{i}softmax'] = nn.Softmax(dim=2)

            module = nn.Sequential(components)
            self.modules_list.append(module)

    def forward(self,
                reward_generators: list[RewardGenerator],
                device: str):
        """
        The forward method gets a list of RewardGenerator objects, rather than
        a tensor of input values. The reward generators calculate a reward
        based on the lever pulled by each module in self.modules_list.
        Input tensors are constructed from random floats in the range [0, 1],
        resulting in a random first lever pull.
        :param reward_generators: A list of RewardGenerator methods.
        :param device:
        :return:
        """
        # We will store rewards and lever choices in lists and return them as
        # well, to facilitate analysis of data. Each row is specific to a
        # RewardGenerator in reward_generators.
        rewards = list()
        levers = list()

        # We initialize an input tensor to random values in the range [0, 1]
        # len(reward_generators) is our batch size.
        module_input = torch.rand(
            (len(reward_generators), self.n), requires_grad=True
        ).to(device)

        # We iterate through each of the input tensors, calculating which
        # lever is pulled and the resulting reward. We append these results
        # to all_rewards and levers. This is our initialization step to
        # create our input for our first module.
        for i, x in enumerate(module_input):
            max_indices = list((x == torch.max(x)).nonzero(as_tuple=True)[0])
            # There is a very small chance of getting two identical values in
            # our input vector, so we invoke np.random.choice to deal with
            # this possibility.
            lever = np.random.choice(max_indices)
            levers.append([lever])
            reward = reward_generators[i].get_reward(lever)
            rewards.append([reward])

        # Our model expects inputs in the shape: (batch size, 1, num levers + 1)
        # We need "num levers + 1" because we also feed the reward value from
        # the previous layer's output into each module.
        module_input = module_input.reshape(
            [len(reward_generators), 1, self.n]
        )
        # We can use our rewards list here. When we iterate through the
        # modules below, we will need to locally track them in a
        # local_rewards list because rewards will accumulate multiple columns
        # of outputs.
        local_rewards = torch.Tensor(rewards).reshape(
            [len(reward_generators), 1, 1]
        ).to(device)
        module_input = torch.cat(
            [module_input, local_rewards], dim=2
        ).to(device)

        # module_outputs will accumulate the output layers of the modules. We
        # will return this Tensor, and our loss and optimization methods will
        # use it to modify parameter weights. Each row of this tensor will be
        # specific to a RewardGenerator in reward_generators.
        module_outputs = None
        # We initialize module_output to be None to facilitate identification
        # of our first pass.
        module_output = None

        # We iterate through each module in our list of modules, stacking the
        # output of each module onto it's input to create the input for the
        # next module.
        for pull, module in enumerate(self.modules_list):
            # After the first loop, we will want to concatenate our output to
            # our previous module's input to generate our new input for the
            # next module.
            if module_output is not None:  # First pass only
                module_input = \
                    torch.cat((module_input, module_output), dim=2).to(device)
            # Do a forward pass through the module.
            module_output = module(module_input)

            # For each RewardGenerator in the batch, determine the lever
            # pulled and the reward generated for pulling the lever.
            local_rewards = list()
            for i, x in enumerate(module_output):
                max_indices = list(torch.ravel(
                    (x == torch.max(x)).nonzero(as_tuple=True)[1]
                ))
                # There is a very small chance of getting two identical values
                # in our output vector, so we invoke np.random.choice to deal
                # with this possibility.
                lever = np.random.choice(max_indices)
                levers[i].append(lever)
                reward = reward_generators[i].get_reward(lever)
                rewards[i].append(reward)
                local_rewards.append([reward])

            # We append module_output to module_outputs BEFORE tacking the
            # reward values on, since we won't be back-propagating through
            # the rewards, just the module outputs.
            if module_outputs is None:  # First pass only
                module_outputs = module_output
            else:
                module_outputs = torch.cat(
                    [module_outputs, module_output], dim=2
                )
            local_rewards = torch.Tensor(local_rewards).reshape(
                [len(reward_generators), 1, 1]
            ).to(device)
            module_output = torch.cat(
                [module_output, local_rewards], dim=2
            ).to(device)

        return module_outputs, levers, rewards
