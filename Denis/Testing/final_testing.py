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


def get_model_rewards(
        model: MABInceptionModel2,
        reward_generators: list[RewardGeneratorChallenging]
):




if __name__ == '__main__':
    main()
