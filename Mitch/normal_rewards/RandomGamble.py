from Heuristic import *

    
class RandomGamble(Heuristic):
    """Randomly pick arms to play."""
    def __init__(self, n_trials, n_runs, n_arms, sigma):
        super().__init__(n_trials, n_runs, n_arms, sigma)

    def gamble(self):
        self.rewards = []
        self.played_optimal = []
        for _ in range(self.n_runs):
            i_t = random.choice(self.arms)
            self.is_optimal(i_t)
            r_t = self.play_arm(i_t)
            self.rewards.append(r_t)
        self.rewards = list(itertools.chain(*self.rewards))
        self.optimal_plays = np.array(self.played_optimal).cumsum()
        self.percent_optimal = [i/(idx+1) for idx,i in enumerate(self.optimal_plays)]

        self.optimal_rewards = self.arm2pull[self.best_arm[0]].rvs(size=self.n_runs)
        self.regrets = self.optimal_rewards - np.array(self.rewards)


if __name__ == '__main__':
    n_arms = 3
    n_trials = 100
    n_runs = 10
    sigma = 0.01
    rg = RandomGamble(n_trials, n_runs, n_arms, sigma)
    rg.simulate_trials()

    plt.subplot(1,2,1)
    plt.plot(rg.avg_regret_per_turn)
    plt.xlabel("Regret per turn")
    
    plt.subplot(1,2,2)
    plt.plot(rg.avg_percent_optimal)
    plt.xlabel("Percentage of optimal arm plays")
    plt.show();