from __future__ import annotations

import os
import pickle
import sys
from typing import Type, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from Denis.nn_models import MABInceptionModel2
from Denis.reward_generators import RewardGenerator, RewardGeneratorChallenging

PICKLED_DIR = "pickled_data"


def main():
    dttm_started = pd.Timestamp.now()
    print(f"Started at {dttm_started}...")

    get_squished_test_gens()

    dttm_finished = pd.Timestamp.now()
    print(f"Finished at {dttm_finished}...")
    print(f"Elapsed time: {dttm_finished-dttm_started}")
    sys.exit(0)


def get_baseline_generators(
        generator: Type[RewardGenerator],
        kwargs: dict[str, Any],
        num_gens: int = 10_000,
        seed: int = 666,
        use_pickled: bool | str = True
):
    pickled_path = (
        f"baseline_gen_set_"
        f"{generator.__name__}_"
        f"{num_gens}_"
        f"{seed}"
    )
    for arg, value in kwargs.items():
        pickled_path += f"_{arg}_{value}"
    pickled_path += ".pickle"
    pickled_path = os.path.join(
        PICKLED_DIR, pickled_path
    )

    if os.path.exists(pickled_path) and use_pickled:
        with open(pickled_path, 'rb') as file:
            reward_gens = pickle.load(file)

    else:
        np.random.seed(seed)
        reward_gens = [
            generator(**kwargs) for _ in range(num_gens)
        ]
        with open(pickled_path, 'wb') as file:
            pickle.dump(reward_gens, file)

    return reward_gens


def get_squished_test_gens():
    """
    This returns our standard test generators for a neural network trained on
    the RewardGeneratorChallenging generator with default values. Note that
    it does not generator generators with the default values, but rather with
    a narrow range of values, to prevent the trained network from using it's
    knowledge of the extremes of the distribution to improve performance.
    :return:
    """
    gens = get_baseline_generators(
        RewardGeneratorChallenging,
        {
            'range_mult_low': 2.5,
            'range_mult_high': 7.5
        }
    )
    return gens


if __name__ == '__main__':
    main()
