from Heuristic import *

class GreedyGamble(Heuristic):
    """Pick arms based on which has the highest expected value."""
    def __init__(self, n_trials, n_runs, n_arms, sigma):
        super().__init__(n_trials, n_runs, n_arms, sigma)

    def gamble(self):
        self.rewards = []
        self.played_optimal = []
        self.arm2sum = {a:0 for a in self.arms}
        self.arm2num = {a:0 for a in self.arms}
        for i_t in self.arms:
            self.is_optimal(i_t)
            r_t = self.play_arm(i_t)
            self.rewards.append(r_t)
            self.arm2sum[i_t] += r_t
            self.arm2num[i_t] += 1
        for _ in range(len(self.arms), self.n_runs):
            arm2mu_est = {a:self.arm2sum[a]/self.arm2num[a] for a in self.arms if self.arm2num[a]>0}
            best = max(arm2mu_est.values())
            best_arms = [a for a in self.arms if arm2mu_est[a] == best]
            i_t = random.choice(best_arms)

            self.is_optimal(i_t)
            r_t = self.play_arm(i_t)
            self.rewards.append(r_t)

        self.rewards = list(itertools.chain(*self.rewards))
        self.optimal_plays = np.array(self.played_optimal).cumsum()
        self.percent_optimal = [i/(idx+1) for idx,i in enumerate(self.optimal_plays)]

        self.optimal_rewards = self.arm2pull[self.best_arm[0]].rvs(size=self.n_runs)
        self.regrets = self.optimal_rewards - np.array(self.rewards)


if __name__ == '__main__':
    n_arms=3
    n_trials=10
    n_runs=10
    sigma=0.01
    gg = GreedyGamble(n_trials, n_runs, n_arms, sigma)
    gg.simulate_trials()

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
    ax1.plot(gg.avg_regret_per_turn)
    ax2.plot(gg.avg_percent_optimal)
    ax1.set_title('Regret per turn')
    ax2.set_title('Percentage of optimal arm plays')
    plt.show();