from funcs import *

n_arms = [2,5,10,50]
n_trials = 1000
n_runs = 1000
m = 5
sigma = [0.01, 0.1, 1]
epsilon = 0.05


for arms in n_arms:
    assert arms*m < n_runs, "(arms*m) must be less than n_trials"
    for s in sigma:
        gg, eg, ef, ucb = init_heuristics(n_trials, n_runs, arms, s, epsilon, m)
        # Save data
        for heuristic in [gg, eg, ef, ucb]:
            h = str(heuristic).split('.')[0].split('<')[1]
            np.savetxt(f"data/{h}-regret-matrix_{arms}arms_{s}sigma.txt", heuristic.regret_matrix)
            np.savetxt(f"data/{h}-optimal-play-matrix_{arms}arms_{s}sigma.txt", heuristic.optimal_play_matrix)
        plot_heuristics(gg, eg, ef, ucb, s)
        plt.savefig(f"img/{arms}-arms_{s}-sigma.png")

# Load data
arm = 2
sigma = 0.1
regret_data, optimal_data = data_loader(arm, sigma)
plt.plot(regret_data['GreedyGamble'].mean(axis=0))
plt.show();