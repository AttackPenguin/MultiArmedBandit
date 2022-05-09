from Heuristic import *

class EpsilonFirstGreedy(Heuristic):
    def __init__(self, n_trials, n_runs, n_arms, sigma, m):
        super().__init__(n_trials, n_runs, n_arms, sigma)
        self.m = m
    
    def gamble(self):
        self.rewards = []
        self.played_optimal = []
        self.arm2sum = {a:0 for a in self.arms}
        self.arm2num = {a:0 for a in self.arms}
        for i_t in self.arms:
            for _ in range(self.m):
                self.is_optimal(i_t)
            r_t = self.play_arm(i_t, self.m)
            self.rewards.append(r_t)
            self.arm2sum[i_t] += r_t.sum()
            self.arm2num[i_t] += self.m
        for _ in range(len(self.arms)*self.m, self.n_runs):
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
    n_arms = 2
    n_trials = 100
    n_runs = 100
    sigma = 0.01
    m = 5
    ef = EpsilonFirstGreedy(n_trials, n_runs, n_arms, sigma, m)
    ef.simulate_trials()

    plt.subplot(1,2,1)
    plt.plot(ef.avg_regret_per_turn)
    plt.xlabel("Regret per turn")
    
    plt.subplot(1,2,2)
    plt.plot(ef.avg_percent_optimal)
    plt.xlabel("Percentage of optimal arm plays")
    plt.show();