from __future__ import annotations

import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def main():
    g_training_loss(
        os.path.join(
            "/home/denis/PycharmProjects/MultiArmedBandit/"
            "Denis/Experiment 01/2022-02-16 14:55:55"
        )
    )


def g_training_loss(
        dir: str,
        stored_weights: bool = True,
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

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    bw_indices = [
        bw_locations.index(value)
        for value in bw_locations
        if start <= value <= end
    ]

    ax.plot(
        range(start+1, end+1),
        losses(start, end)
    )
    ax.plot(
        [index+1 for index in bw_indices],
        [bw_locations[index] for index in bw_indices]
    )

    fig.show()


if __name__ == '__main__':
    main()