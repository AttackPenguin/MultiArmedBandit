import numpy as np


class RewardGenerator01:
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
                   n: int):
        """
        :param n: "Lever" to pull.
        :return: A value from a normal distribution with a mean of
        self.means[n] and a std deviation of self.std. Truncated to be in the
        range [0, 1].
        """
        if n < 0 or n > self.n:
            raise ValueError(f"n is {n}, must be in range [0, {self.n}].")
        return_val = np.random.normal(
            loc=self.means[n],
            scale=self.std
        )
        # Check to ensure is in range [0, 1].
        if return_val < 0:
            return 0.0
        if return_val > 1:
            return 1.0
        return return_val

    def get_max_mean(self):
        """
        :return: The maximum mean in the distributions of return values.
        """
        return max(self.means)

    def get_best_lever(self):
        """
        :return: The index of the maximum distribution mean.
        """
        return self.means.index(max(self.means))
