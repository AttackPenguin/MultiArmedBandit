from __future__ import annotations

from abc import abstractmethod

import numpy as np
from scipy import stats


class RewardGenerator:
    @abstractmethod
    def __init__(self,
                 is_subclass = False):
        pass

    @abstractmethod
    def get_reward(self,
                   n: int) -> int | float:
        pass

    @abstractmethod
    def get_max_mean(self) -> int | float:
        pass

    @abstractmethod
    def get_best_lever(self) -> int:
        pass


class RewardGenerator01(RewardGenerator):
    """
    Randomly generates a list of n values in the range [0, 1]. These values
    are used as the means of normal distributions with standard deviations of
    std. Has methods to retrieve values from each distribution, to retrieve
    the highest mean, and to retrieve the 'lever' associated with the highest
    mean.
    Return values are truncated to be in the range [0, 1]. When values higher
    than this range are generated, 1 is returned, and when values lower than
    this range are generated, 0 is returned.
    """
    def __init__(self,
                 n: int = 10,
                 std: float = 0.1):
        """
        :param n: The number of distributions to create.
        :param std: The standard deviation of the distributions.
        """
        self.n = n
        self.std = std
        # Means are stored in a list and accessed via their index.
        self.means = list()
        for _ in range(n):
            mean = np.random.rand()
            self.means.append(mean)

    def get_reward(self,
                   n: int) -> int | float:
        """
        :param n: "Lever" to pull.
        :return: A value from a normal distribution with a mean of
        self.means[n] and a std deviation of self.std. Truncated to be in the
        range [0, 1].
        """
        if n < 0 or n >= self.n:
            raise ValueError(f"n is {n}, must be in range [0, {self.n-1}].")
        # Use a bounded normal distribution
        lower, upper = 0, 1
        return_val = stats.truncnorm(
            (lower - self.means[n]) / self.std,
            (upper - self.means[n]) / self.std,
            loc = self.means[n],
            scale = self.std
        )
        return return_val.rvs(1)

    def get_max_mean(self) -> int | float:
        """
        :return: The maximum mean in the distributions of return values.
        """
        return max(self.means)

    def get_best_lever(self) -> int:
        """
        :return: The index of the maximum distribution mean.
        """
        return self.means.index(max(self.means))
