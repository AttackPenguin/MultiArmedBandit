from __future__ import annotations

import os
import pickle
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from Denis.Testing.testing_generators import get_baseline_generators
from Denis.nn_models import MABInceptionModel2
from Denis.reward_generators import RewardGeneratorChallenging

PICKLED_DIR = "pickled_data"

sns.set()


def main():
    pass


def get_model_rewards(
        model: MABInceptionModel2,
        reward_generators: list[RewardGeneratorChallenging]
):
    rewards = list()
    for i in range(0, len(reward_generators), 1000):
        _, _, local_rewards = model(reward_generators[i:i+1000])
        rewards += local_rewards



if __name__ == '__main__':
    main()
