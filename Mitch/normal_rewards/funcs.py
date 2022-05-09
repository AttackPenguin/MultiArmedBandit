from Heuristic import *
from RandomGamble import RandomGamble
from GreedyGamble import GreedyGamble
from EpsilonGreedy import EpsilonGreedy
from EpsilonFirstGreedy import EpsilonFirstGreedy
from UCB import UCB

def init_heuristics(n_trials, n_runs, n_arms, sigma, epsilon, m):
    """Initialize heuristic functions."""
    # rg = RandomGamble(n_trials, n_runs, n_arms, sigma)
    # rg.simulate_trials()

    gg = GreedyGamble(n_trials, n_runs, n_arms, sigma)
    gg.simulate_trials()

    eg = EpsilonGreedy(n_trials, n_runs, n_arms, sigma, epsilon)
    eg.simulate_trials()

    ef = EpsilonFirstGreedy(n_trials, n_runs, n_arms, sigma, m)
    ef.simulate_trials()

    ucb = UCB(n_trials, n_runs, n_arms, sigma)
    ucb.simulate_trials()

    return gg, eg, ef, ucb


def plot_heuristics(gg, eg, ef, ucb, sigma):
    """Plot regret per turn and percentage of optimal plays."""
    plt.figure(figsize=(12,6))
    plt.suptitle(f"$\sigma=${sigma}")
    plt.subplot(1,2,1)
    plt.xlabel("Regret per turn")
    # plt.plot(rg.avg_regret_per_turn, label="Random Gamble")
    plt.plot(gg.avg_regret_per_turn, label="Greedy Gamble")
    plt.plot(eg.avg_regret_per_turn, label="Epsilon Greedy")
    plt.plot(ef.avg_regret_per_turn, label="Epsilon First Greedy")
    plt.plot(ucb.avg_regret_per_turn, label="UCB")
    plt.legend(bbox_to_anchor=(0,1.02, 2.2, .102), loc='lower left', 
                ncol=5, mode="expand", borderaxespad=0)

    plt.subplot(1,2,2)
    plt.xlabel("Percentage of optimal arm")
    # plt.plot(rg.avg_percent_optimal)
    plt.plot(gg.avg_percent_optimal)
    plt.plot(eg.avg_percent_optimal)
    plt.plot(ef.avg_percent_optimal)
    plt.plot(ucb.avg_percent_optimal)


def data_loader(arm, sigma):
    """Load data from heuristic simulations."""
    heuristics = "GreedyGamble EpsilonGreedy EpsilonFirstGreedy UCB".split()
    regret_data = {h:[] for h in heuristics}
    optimal_data = {h:[] for h in heuristics}
    for h in regret_data:
        regret_data[h] = np.loadtxt(f"data/{h}-regret-matrix_{arm}arms_{sigma}sigma.txt", dtype=float)
        optimal_data[h] = np.loadtxt(f"data/{h}-optimal-play-matrix_{arm}arms_{sigma}sigma.txt", dtype=float)
    return regret_data, optimal_data