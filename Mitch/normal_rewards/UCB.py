from Heuristic import *

class UCB(Heuristic):
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

        for t in range(len(self.arms), self.n_runs):
            arm2mu_est = {a:self.arm2sum[a]/self.arm2num[a] for a in self.arms if self.arm2num[a]>0}
            arm2ucb = {a:(r + np.sqrt((2*np.log(t)) / self.arm2num[a])) for a,r in arm2mu_est.items()}
            best = max(arm2ucb.values())
            best_arms = [a for a in self.arms if arm2ucb[a] == best]
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
    n_arms = 3
    n_trials = 100
    n_runs = 10
    sigma = 0.01
    ucb = UCB(n_trials, n_runs, n_arms, sigma)
    ucb.simulate_trials()

    plt.subplot(1,2,1)
    plt.plot(ucb.avg_regret_per_turn)
    plt.xlabel("Regret per turn")
    
    plt.subplot(1,2,2)
    plt.plot(ucb.avg_percent_optimal)
    plt.xlabel("Percentage of optimal arm plays")
    plt.show();