import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class RewardGenerator:
    def __init__(self,
                 n: int = 10,
                 std: float = 0.1):
        self.n = n
        self.std = std

        self.means = list()
        for _ in range(n):
            mean = np.random.rand()
            self.means.append(mean)

    def get_reward(self,
                   n: int):

        if n < 0 or n > self.n:
            raise ValueError(f"n is {n}, must be in range [0, {self.n}].")
        return_val = np.random.normal(
            loc=self.means[n],
            scale=self.std
        )
        if return_val < 0:
            return 0.0
        if return_val > 1:
            return 1.0
        return return_val

    def get_max_mean(self):
        return max(self.means)

    def get_best_lever(self):
        return self.means.index(max(self.means))


class MultiArmedBandit(nn.Module):
    def __init__(self,
                 n: int = 10,
                 pulls: int = 100):
        super(MultiArmedBandit, self).__init__()
        # We will store a list of modules that will pull n-1 levers. Our
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
                nn.Softmax(dim=2)
            )
            self.modules_list.append(module)

    def forward(self,
                reward_generators: list[RewardGenerator]):
        module_outputs = list()
        all_rewards = list()

        module_input = torch.rand((len(reward_generators), self.n))
        # add the lever pull output to each sub-tensor
        rewards = list()
        for gen_index, x in enumerate(module_input):
            lever = int((x == torch.max(x)).nonzero(as_tuple=True)[0])
            module_outputs.append([x.reshape((1, 10))])
            reward = reward_generators[gen_index].get_reward(lever)
            rewards.append([reward])
            all_rewards.append([reward])
        module_input = module_input.reshape(
            [len(reward_generators), 1, self.n]
        )
        rewards = torch.Tensor(rewards).reshape(2, 1, 1)
        module_input = torch.cat(
            [module_input, rewards], dim=2
        )
        module_output = None

        for pull, module in enumerate(self.modules_list):
            if module_output is not None:
                module_input = torch.cat((module_input, module_output), dim=2)
            module_output = module(module_input)
            # add the lever pull output to each sub-tensor
            rewards = list()
            for gen_index, x in enumerate(module_output):
                module_outputs[gen_index].append(x)
                max_indices = list(torch.ravel(
                    (x == torch.max(x)).nonzero(as_tuple=True)[1]
                ))
                lever = np.random.choice(max_indices)
                reward = reward_generators[gen_index].get_reward(lever)
                rewards.append([reward])
                all_rewards[gen_index].append(reward)
            rewards = torch.Tensor(rewards).reshape(2, 1, 1)
            module_output = torch.cat(
                [module_output, rewards], dim=2
            )

        return module_outputs, rewards


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n\n")

model = MultiArmedBandit()
# print(model)

reward_gens = [RewardGenerator(), RewardGenerator()]
module_outputs, rewards = model(reward_gens)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


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
          batch_size: int = 256,
          num_data_loader_workers: int = 4,
          training_rounds: int = 1024*1024):

    dataset = RewardsGeneratorDataset(size=training_rounds)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_data_loader_workers)

    for batch, reward_gens in enumerate(dataloader):
        module_outputs, best_outputs = model(reward_gens)


