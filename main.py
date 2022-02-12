import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class RewardGenerator:
    """
    Randomly generates a list of n values in the range [0, 1]. These values
    are used as the means of normal distributions with standard deviations of
    std. Has methods to retrieve values from each distribution, to retrieve
    the highest mean, and to retrieve the 'lever' associated with the highest
    mean.
    Return values are truncated to be in the range [0, 1]. When values higher
    than this range are generated, 1 is returned, and when values lower than
    this range are generated, 0 is returned.
    """
    def __init__(self,
                 n: int = 10,
                 std: float = 0.1):
        """
        :param n: The number of distributions to create.
        :param std: The standard deviation of the distributions.
        """
        self.n = n
        self.std = std
        # Means are stored in a list and accessed via their index.
        self.means = list()
        for _ in range(n):
            mean = np.random.rand()
            self.means.append(mean)

    def get_reward(self,
                   n: int):
        """
        :param n: "Lever" to pull.
        :return: A value from a normal distribution with a mean of
        self.means[n] and a std deviation of self.std. Truncated to be in the
        range [0, 1].
        """
        if n < 0 or n > self.n:
            raise ValueError(f"n is {n}, must be in range [0, {self.n}].")
        return_val = np.random.normal(
            loc=self.means[n],
            scale=self.std
        )
        # Check to ensure is in range [0, 1].
        if return_val < 0:
            return 0.0
        if return_val > 1:
            return 1.0
        return return_val

    def get_max_mean(self):
        """
        :return: The maximum mean in the distributions of return values.
        """
        return max(self.means)

    def get_best_lever(self):
        """
        :return: The index of the maximum distribution mean.
        """
        return self.means.index(max(self.means))


class MultiArmedBandit(nn.Module):
    def __init__(self,
                 n: int = 10,
                 pulls: int = 100):
        super(MultiArmedBandit, self).__init__()
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
                nn.Linear(i * (n + 1), 256),
                nn.RReLU(),
                nn.Dropout(),
                nn.Linear(256, 256),
                nn.RReLU(),
                nn.Dropout(),
                nn.Linear(256, 256),
                nn.RReLU(),
                nn.Dropout(),
                nn.Linear(256, n),
                nn.RReLU(),
                # A dropout layer here causes a problem in which the softmax
                # layer would often output multiple identical values in early
                # forward passes. Uncertain why. For now leaving it out.
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
            lever = int((x == torch.max(x)).nonzero(as_tuple=True)[0])
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


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n\n")

model = MultiArmedBandit()
# print(model)

# Forward Method Test Code:
# reward_gens = [RewardGenerator(), RewardGenerator()]
# module_outputs, rewards = model(reward_gens)



class RewardsGeneratorDataset(Dataset):
    def __init__(self,
                 n: int = 10,
                 pulls: int = 100,
                 std: float = 0.1,
                 size: int = 1024 * 1024):
        self.n = n
        self.std = std
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        reward_generator = RewardGenerator(self.n, self.std)
        best_lever = reward_generator.get_best_lever()
        best_outputs = torch.zeros((1, self.n))
        best_outputs = torch.cat([best_outputs]*self.pulls, dim=1)
        return reward_generator


def train(model: nn.Module,
          loss_fn: callable,
          optimizer: torch.optim.Optimizer,
          n: int = 10,
          pulls: int = 100,
          batch_size: int = 256,
          num_data_loader_workers: int = 4,
          training_rounds: int = 1000):

    # dataset = RewardsGeneratorDataset(size=training_rounds)
    # dataloader = DataLoader(dataset,
    #                         batch_size=batch_size,
    #                         num_workers=num_data_loader_workers)
    #
    # for i, reward_gens in enumerate(dataloader):
    #     module_outputs, levers, rewards = model(reward_gens)
    #     # Build tensor of optimal choices
    #     optimal_outputs = None
    #     for gen in reward_gens:
    #         pass
    model.train(True)
    for i in range(1, training_rounds+1):
        reward_gens = [
            RewardGenerator() for i in range(batch_size)
        ]

        optimal_lever = reward_gens[0].get_best_lever()
        optimal_output = [0]*n
        optimal_output[optimal_lever] = 1
        optimal_output = optimal_output * (pulls-1)
        optimal_output = torch.Tensor(optimal_output)
        optimal_outputs = optimal_output.reshape(
            (1, 1, len(optimal_output))
        )
        for j in range(1, len(reward_gens)):
            optimal_lever = reward_gens[j].get_best_lever()
            optimal_output = [0]*n
            optimal_output[optimal_lever] = 1
            optimal_output = optimal_output * (pulls-1)
            optimal_output = torch.Tensor(optimal_output)
            optimal_output = optimal_output.reshape(
                (1, 1, len(optimal_output))
            )
            optimal_outputs = torch.cat(
                (optimal_outputs, optimal_output), dim=0
            )

        module_outputs, levers, rewards = model(reward_gens)
        loss = loss_fn(module_outputs, optimal_outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(f"{i} iterations complete.")


# train method test code:
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
model.to(device)
train(model, loss_fn, optimizer)

model.train(False)

reward_totals = list()
for _ in range(1000):
    gen = RewardGenerator()
    _, _, rewards = model([gen])
    reward_totals.append(sum(rewards[0]))
    # print(sum(rewards[0]))

print(np.mean(reward_totals))
