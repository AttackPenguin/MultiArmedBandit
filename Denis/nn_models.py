import numpy as np
import torch
from torch import nn

from Denis.main import RewardGenerator


class MABInceptionModel(nn.Module):
    def __init__(self,
                 n: int = 10,
                 pulls: int = 100,
                 module_width: int = 256):
        super(MABInceptionModel, self).__init__()
        # We will store a list of modules that will pull (pulls-1) levers. Our
        # first lever pull will be random, so there is no reason to
        # implement a module for it.
        self.n = n
        self.modules_list = nn.ModuleList()
        for i in range(1, pulls):
            module = nn.Sequential(
                # Each module takes the output of all prior modules,
                # plus their calculated rewards, so we increase the expected
                # input size as we add modules.
                nn.Linear(i * (n + 1), module_width),
                nn.RReLU(),
                nn.Dropout(),
                nn.Linear(module_width, module_width),
                nn.RReLU(),
                nn.Dropout(),
                nn.Linear(module_width, module_width),
                nn.RReLU(),
                nn.Dropout(),
                nn.Linear(module_width, n),
                nn.RReLU(),
                nn.Softmax(dim=2)
            )
            self.modules_list.append(module)

    def forward(self,
                reward_generators: list[RewardGenerator]):
        """
        The forward method gets a list of methods that calculate rewards
        based on which levers are pulled, rather than a tensor of input
        values. Input values are random floats in the range [0, 1], resulting
        in a random first lever pull.
        :param reward_generators: A list of RewardGenerator methods.
        :return:
        """
        # We will store rewards and lever choices in lists and return them as
        # well, to facilitate analysis of data. Each row is specific to a
        # RewardGenerator in reward_generators.
        rewards = list()
        levers = list()

        # We initialize an input tensor to random values in the range [0, 1]
        module_input = torch.rand(
            (len(reward_generators), self.n), requires_grad=True
        )

        # We iterate through each of the input tensors, calculating which
        # lever is pulled and the resulting reward. We append these results
        # to all_rewards and levers.
        for i, x in enumerate(module_input):
            max_indices = list((x == torch.max(x)).nonzero(as_tuple=True)[0])
            lever = np.random.choice(max_indices)
            # lever = int((x == torch.max(x)).nonzero(as_tuple=True)[0])
            levers.append([lever])
            reward = reward_generators[i].get_reward(lever)
            rewards.append([reward])

        # Our model expects inputs in the shape: (batch size, 1, num levers + 1)
        # We need "num levers + 1" because we also feed the reward value from
        # the previous layer's output into each module.
        module_input = module_input.reshape(
            [len(reward_generators), 1, self.n]
        )
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
            if module_output is not None:
                module_input = torch.cat((module_input, module_output), dim=2)
            # Do a forward pass through the module.
            module_output = module(module_input)
            # Add the lever pull output to each sub-tensor
            local_rewards = list()
            local_module_outputs = list()
            for i, x in enumerate(module_output):
                local_module_outputs.append(x)
                max_indices = list(torch.ravel(
                    (x == torch.max(x)).nonzero(as_tuple=True)[1]
                ))
                lever = np.random.choice(max_indices)
                levers[i].append(lever)
                reward = reward_generators[i].get_reward(lever)
                rewards[i].append(reward)
                local_rewards.append([reward])
            # We append module_output to module_outputs BEFORE tacking the
            # reward values on, since we won't be back-propagating through them.
            if module_outputs is None:
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
