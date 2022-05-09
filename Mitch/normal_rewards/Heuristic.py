import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy, scipy.stats


class Heuristic:
    def __init__(self, n_trials, n_runs, n_arms, sigma):
        self.n_trials = n_trials
        self.n_runs = n_runs
        self.n_arms = n_arms
        self.sigma = sigma

        self.arm2pull = {}
        self.arm2trueExpectedReward = {}
        self.run2regrets = {}
        self.trunc_norm()

    def trunc_norm(self):
        """Generate reward distributions. Mean is selected uniformly from [0,1]
        and rewards are generated from a truncated normal distribution.
        """
        lower_bound = 0
        upper_bound = 1
        mus = scipy.stats.uniform.rvs(loc=lower_bound, scale=upper_bound, size=self.n_arms)
        for arm,mu in enumerate(mus):
            a,b = (lower_bound - mu)/self.sigma, (upper_bound-mu)/self.sigma
            p = scipy.stats.truncnorm(a,b,loc=mu,scale=self.sigma)
            self.arm2pull[arm] = p
            self.arm2trueExpectedReward[arm] = mu
        self.arms = list(self.arm2pull.keys())
        self.mu_star = max(self.arm2trueExpectedReward.values())
        self.best_arm = [i for i in self.arm2trueExpectedReward.keys() if self.arm2trueExpectedReward[i] == self.mu_star]

    def play_arm(self, i_t, n=1):
        """ Get the reward from playing arm i_t, n times.  """
        return self.arm2pull[i_t].rvs(size=n)
    
    def is_optimal(self, i_t):
        """Determine if lever pull was optimal play."""
        if i_t in self.best_arm:
            self.played_optimal.append(1)
        else:
            self.played_optimal.append(0)
    
    def simulate_trials(self):
        self.regret_matrix = np.zeros((self.n_trials, self.n_runs))
        self.optimal_play_matrix = np.zeros((self.n_trials, self.n_runs))
        for run in range(self.n_trials):
            self.trunc_norm()
            self.gamble()
            self.regret_matrix[run] = self.regrets
            self.optimal_play_matrix[run] = self.percent_optimal
        self.avg_regret_per_turn = self.regret_matrix.mean(axis=0)
        self.avg_percent_optimal = self.optimal_play_matrix.mean(axis=0)
    

    

