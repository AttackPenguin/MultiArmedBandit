from __future__ import annotations

from abc import abstractmethod

import numpy as np
from scipy import stats


class RewardGenerator:
    @abstractmethod
    def __init__(self):
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


class RewardGeneratorTruncNorm(RewardGenerator):
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
    def __init__(self, n: int = 10, std: float = 0.1):
        """
        :param n: The number of distributions to create.
        :param std: The standard deviation of the distributions.
        """
        super().__init__()
        self.n = n
        self.std = std
        # Means are stored in a list and accessed via their index.
        self.means = list()
        for _ in range(n):
            mean = float(np.random.rand())
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
        return float(return_val.rvs(1))

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


class RewardGeneratorChallenging(RewardGenerator):
    """


    """
    def __init__(self,
                 n: int = 10,
                 range_mult_low: int = 1,
                 range_mult_high: int = 10):
        """
        :param n: The number of distributions to create.
        :param std: The standard deviation of the distributions.
        """
        super().__init__()
        self.n = n

        # We will use an exponential distribution to pick the minimum and
        # maximum values of the range of possible rewards.
        # We use a range multiplier here on one value so that if the neural
        # network learns what constitutes a 'high' value in the distribution
        # we can cripple that learning by testing it on a dataset with a
        # smaller range_multiplier, which will result in reward generators
        # which return smaller values.
        expon_dist = stats.expon()
        values = [
            expon_dist.rvs(),
            expon_dist.rvs()]
        self.min_value = min(values) * range_mult_low
        self.max_value = max(values) * range_mult_high

        # Randomly generate alpha and beta values for beta distributions
        # attached to levers. Store both in lists. Levers will be specified
        # by reference to list indices. Values will be generated from an
        # exponential distribution, multiplied by 10 to increase the mean
        # well above 1 and make beta functions in which bother variables are
        # under 1 occur rare.
        self.alphas = list()
        self.betas = list()
        self.means = list()
        for _ in range(n):
            self.alphas.append(expon_dist.rvs() * 10)
            self.betas.append(expon_dist.rvs() * 10)
            self.means.append(
                stats.beta(self.alphas[-1], self.betas[-1]).mean() *
                (self.max_value - self.min_value) +
                self.min_value
            )

    def get_reward(self,
                   n: int) -> int | float:
        """
        :param n: "Lever" to pull.
        :return: Returns a value from a beta distribution that has been
        shifted and stretched to match the calculated minimum and maximum
        values.
        """
        if n < 0 or n >= self.n:
            raise ValueError(f"n is {n}, must be in range [0, {self.n-1}].")

        return (
            stats.beta(self.alphas[n], self.betas[n]).rvs() *
            (self.max_value - self.min_value) +
            self.min_value
        )

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
