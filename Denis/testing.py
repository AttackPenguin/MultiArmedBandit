from __future__ import annotations

import os
import pickle
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from Denis.nn_models import MABInceptionModel
from Denis.reward_generators import RewardGenerator, RewardGeneratorTruncNorm

sns.set()


def main():
    g_training_loss(
        os.path.join(
            "/home/denis/PycharmProjects/MultiArmedBandit/"
            "Denis/Experiment 02/2022-03-04 15:27:08"
        ),
        show_best_weights=True,
        start=None,
        end=None
    )

    # gens = get_baseline_generators(
    #     RewardGeneratorTruncNorm
    # )
    # levers, rewards, total_rewards, opt_total_rewards = \
    #     get_reward_data_from_nn(
    #         "/home/denis/PycharmProjects/MultiArmedBandit/"
    #         "Denis/Experiment 01/2022-02-16 14:55:55/"
    #         "model_weights_round_16624_mtr_0.24146.pth",
    #         gens
    #     )
    # plt.hist(
    #     total_rewards,
    #     bins=100,
    #     density=True
    # )
    # plt.show()
    # plt.cla()
    # reward_by_round_t = np.array(rewards).transpose()
    # reward_by_round_mean = np.mean(
    #     reward_by_round_t, axis=1
    # )
    # plt.plot(
    #     range(len(reward_by_round_mean)),
    #     reward_by_round_mean
    # )
    # plt.show()
    pass


def g_training_loss(
        dir: str,
        show_best_weights: bool = True,
        start: int = None,
        end: int = None
):
    """
    Takes directory of experiment as argument, and generates figure showing
    training loss over time.
    :return:
    """

    losses_file_path = os.path.join(
        dir, 'losses.pickle'
    )
    with open(losses_file_path, 'rb') as file:
        losses = pickle.load(file)
    bw_locations_file_path = os.path.join(
        dir, 'best_weights_locations.pickle'
    )
    with open(bw_locations_file_path, 'rb') as file:
        bw_locations = pickle.load(file)
    bw_losses_file_path = os.path.join(
        dir, 'best_weights_losses.pickle'
    )
    with open(bw_losses_file_path, 'rb') as file:
        bw_losses = pickle.load(file)

    if start is None:
        start = 0
    if end is None:
        end = len(losses)

    fig: plt.Figure = plt.figure(figsize=[6.4, 4.6], dpi=400)
    ax: plt.Axes = fig.add_subplot()
    bw_indices = [
        bw_locations.index(value)
        for value in bw_locations
        if start <= value <= end
    ]

    ax.scatter(
        range(start + 1, end + 1),
        losses[start:end],
        s=1,
        alpha=0.2,
        label='All Loss Values'
    )
    if show_best_weights:
        ax.scatter(
            [bw_locations[index] + 1 for index in bw_indices],
            [bw_losses[index] for index in bw_indices],
            s=1,
            color='red',
            label='Saved Weights'
        )
        ax.legend()

    ax.set_title(
        f"Loss Per Round of Training, {dir[-19:]}"
    )
    ax.set_xlabel('Rounds of Training')
    ax.set_ylabel('Loss')

    fig.show()


def get_baseline_generators(
        generator: Type[RewardGenerator],
        num_gens: int = 10_000,
        seed: int = 666,
        load_pickled: bool | str = False
):
    # if load_pickled is False, then we create generators using the
    # seed value. If a string, then treat it as the path to a file,
    # and see if there is a file of pickled generators to return.
    if load_pickled:
        pass  # Needs implementation
    np.random.seed(seed)
    reward_gens = [
        generator() for _ in range(num_gens)
    ]
    return reward_gens


def get_reward_data_from_nn(
        file_path: str,
        generators: list[RewardGenerator],
        model=MABInceptionModel
):
    model = model()
    model.load_state_dict(torch.load(file_path))
    _, levers, rewards = model(generators)
    total_rewards = [
        sum(data) for data in rewards
    ]
    opt_total_rewards = [
        100 * gen.get_max_mean()
        for gen in generators
    ]
    return levers, rewards, total_rewards, opt_total_rewards


def get_reward_data_from_method(
        generators: list[RewardGenerator],
        method: callable
):
    pass


if __name__ == '__main__':
    main()
