from __future__ import annotations

import os
import pickle
from typing import Type

import numpy as np
import pandas as pd
import torch
from torch import nn

from Denis.reward_generators import RewardGenerator


def training_method_01(model: nn.Module,
                       loss_fn: callable,
                       optimizer: torch.optim.Optimizer,
                       reward_generator: Type[RewardGenerator],
                       n: int = 10,
                       pulls: int = 100,
                       batch_size: int = 256,
                       validation_size: int = 100,
                       training_rounds: int = None,
                       save_dir: str = None,
                       validate_interval: int = 5,
                       save_interval: int = 100):

    if save_interval % validate_interval != 0:
        raise ValueError(
            "validate_interval must evenly divide save_interval."
        )

    if os.path.exists(save_dir):
        raise ValueError(
            "Specified save directory already exists. Exiting to avoid "
            "overwriting existing data."
        )
    else:
        os.makedirs(save_dir)

    training_summary = (
        f"Training Started: {pd.Timestamp.now()}\n\n"
        f"Loss Function: {loss_fn.__class__.__name__}\n"
        f"Optimizer: {optimizer.__class__.__name__}\n"
        f"Number of Levers: {n}\n"
        f"Number of Lever Pulls: {pulls}\n"
        f"Batch Size: {batch_size}\n"
        f"Validation Size: {validation_size}\n"
        f"Training Rounds: {training_rounds}\n"
        f"Validation Interval: {validate_interval}\n"
        f"Save Interval: {save_interval}\n\n"
    )
    print('\n' + training_summary)
    training_summary += (
        f"Model Structure: \n\n"
        f"{model}"
    )
    summary_file_path = os.path.join(
        save_dir,
        "Training Configuration.txt"
    )
    with open(summary_file_path, 'w') as f:
        f.write(training_summary)

    if training_rounds is None:
        training_rounds = 1_000_000_000_000_000
        print("Training Rounds not specified. Will train indefinitely. "
              "Performance data will be saved every round. Kill the process "
              "to stop training.")

    rewards_file_path = os.path.join(
        save_dir,
        "mean_total_rewards.pickle"
    )
    best_weights_locs_file_path = os.path.join(
        save_dir,
        "best_weights_locations.pickle"
    )
    best_weights_tot_rew_file_path = os.path.join(
        save_dir,
        "best_weights_tot_rewards.pickle"
    )
    mean_total_rewards = list()
    best_weights = None
    best_weights_location = None
    best_weights_locations = list()
    best_weights_tot_reward = None
    best_weights_tot_rewards = list()

    # We will
    validation_gens = [
        reward_generator() for i in range(validation_size)
    ]

    for i in range(0, training_rounds):
        # Set to training mode.
        model.train(True)

        # Create a batch of reward_generators
        reward_gens = [
            reward_generator() for i in range(batch_size)
        ]

        # Create a tensor of the optimal lever pulls. Each row corresponds to
        # the optimal series of 'pulls' lever pulls of one of 'n' levers. All
        # values of the tensor will be either 1 or 0, with 1 for the optimal
        # lever and 0 for all others.

        # We initially create a tensor for the first reward_generator.
        optimal_lever = reward_gens[0].get_best_lever()
        # List describing a single optimal lever pull:
        optimal_output = [0] * n
        optimal_output[optimal_lever] = 1
        # Multiply by the number of pulls, convert to a tensor,
        # and reshape.
        optimal_output = optimal_output * (pulls - 1)
        optimal_output = torch.Tensor(optimal_output)
        optimal_outputs = optimal_output.reshape(
            (1, 1, len(optimal_output))
        )
        # Now do the same thing for the other reward generators,
        # concatenating each result to the prior results.
        for j in range(1, len(reward_gens)):
            optimal_lever = reward_gens[j].get_best_lever()
            optimal_output = [0] * n
            optimal_output[optimal_lever] = 1
            optimal_output = optimal_output * (pulls - 1)
            optimal_output = torch.Tensor(optimal_output)
            optimal_output = optimal_output.reshape(
                (1, 1, len(optimal_output))
            )
            optimal_outputs = torch.cat(
                (optimal_outputs, optimal_output), dim=0
            )

        # Do a forward pass. We don't care about the lever pulls or rewards
        # when we're optimizing.
        module_outputs, _, _ = model(reward_gens)
        # Loss is the calculated by comparing the optimal pulls to the actual
        # pulls tensors.
        loss = loss_fn(module_outputs, optimal_outputs)
        # And let's optimize.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # If we've reached a validation interval, calculate rewards on
        # validation generators and save data.
        if ((i+1) % validate_interval == 0 or i == training_rounds - 1) \
                and i != 0:

            # Deactivate training mode.
            model.train(False)

            # Get validation rewards
            _, _, rewards = model(validation_gens)
            reward_totals = [
                sum(reward) for reward in rewards
            ]
            mean_total_reward = float(np.mean(reward_totals))

            # Look back through the current window of training rounds to see if
            # we have a new high score. If so, store the current model's
            # weights.
            window = mean_total_rewards[
                     -1 * (i % (save_interval // validate_interval)):
                ]
            if not window:  # Scenario where we've just entered a new window
                best_weights = model.state_dict().copy()
                best_weights_location = i
                best_weights_tot_reward = mean_total_reward
            elif max(window) < mean_total_reward:
                best_weights = model.state_dict().copy()
                best_weights_location = i
                best_weights_tot_reward = mean_total_reward
            mean_total_rewards.append(mean_total_reward)

            # If we've reached the end of a window of training rounds,
            # or if we've completed our final iteration, save the best weights
            # in the window.
            if ((i+1) % save_interval == 0 or i == training_rounds - 1) \
                    and i != 0:
                file_path = os.path.join(
                    save_dir,
                    f"model_weights_round_"
                    f"{best_weights_location}_"
                    f"mtr_"
                    f"{best_weights_tot_reward:.2f}.pth"
                )
                torch.save(best_weights, file_path)
                best_weights = None
                best_weights_locations.append(best_weights_location)
                best_weights_location = None
                best_weights_tot_rewards.append(best_weights_tot_reward)
                print(f"{i} Rounds of Training Completed. "
                      f"Best mean total reward this window: "
                      f"{best_weights_tot_reward:.2f}")
                best_weights_tot_reward = None

                with open(best_weights_locs_file_path, 'wb') as f:
                    pickle.dump(best_weights_locations, f)
                with open(best_weights_tot_rew_file_path, 'wb') as f:
                    pickle.dump(best_weights_tot_rewards, f)

            with open(rewards_file_path, 'wb') as f:
                pickle.dump(mean_total_rewards, f)

