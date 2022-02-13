import numpy as np
import torch
from torch import nn

from Denis.main import RewardGenerator


def train(model: nn.Module,
          loss_fn: callable,
          optimizer: torch.optim.Optimizer,
          n: int = 10,
          pulls: int = 100,
          batch_size: int = 256,
          num_data_loader_workers: int = 4,
          training_rounds: int = 100_000,
          report_modulus: int = 100):

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
    reward_totals = list()
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
        for reward in rewards:
            reward_totals.append(sum(reward))
        if i % report_modulus == 0:
            print(f"{i} iterations complete.")
            print(f"\tMean total reward last {report_modulus*batch_size} "
                  f"generators pushed: "
                  f"{float(np.mean(reward_totals)):.2f}")
            reward_totals = list()
