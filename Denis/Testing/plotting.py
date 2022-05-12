from __future__ import annotations

import os
import pickle
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from Denis.nn_models import MABInceptionModel2
from Denis.reward_generators import RewardGeneratorChallenging

PICKLED_DIR = "pickled_data"

sns.set()


def main():
    g_training_loss(
        "/home/denis/PycharmProjects/MultiArmedBandit/Denis/Experiment 03/2022-03-26 10:09:18",
        True, 14900
    )


def g_training_mean_tot_performance(
        dir_path: str,
        use_pickled: bool = True,
        start: int = None,
        end: int = None
):
    """
    Takes directory of experiment as argument, and generates figure showing
    mean overall performance of stored highest performing model weights over
    time.
    :return:
    """
    pass


def g_training_loss(
        dir_path: str,
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
        dir_path, 'losses.pickle'
    )
    with open(losses_file_path, 'rb') as file:
        losses = pickle.load(file)
    bw_locations_file_path = os.path.join(
        dir_path, 'best_weights_locations.pickle'
    )
    with open(bw_locations_file_path, 'rb') as file:
        bw_locations = pickle.load(file)
    bw_losses_file_path = os.path.join(
        dir_path, 'best_weights_losses.pickle'
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
        f"Loss Per Round of Training, {dir_path[-19:]}"
    )
    ax.set_xlabel('Rounds of Training')
    ax.set_ylabel('Loss')

    fig.show()


if __name__ == '__main__':
    main()
