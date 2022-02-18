from __future__ import annotations

import os
import pickle


def main():
    g_training_loss(
        os.path.join(
            "/home/denis/PycharmProjects/MultiArmedBandit/"
            "Denis/Experiment 01/2022-02-15 20:29:27"
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

    pass


if __name__ == '__main__':
    main()