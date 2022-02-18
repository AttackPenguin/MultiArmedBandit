# HW03
# Mitchell Joseph

"""
Imports.........................lines 20-24

Class MAB
    init()......................lines 28-31
    beta_dist().................lines 33-48
    plot_distribution().........lines 50-65
    play_arm()..................lines 67-75
    gamble()....................lines 77-125
    plot_individual_gamble()....lines 127-135
    plot_average_gamble().......lines 137-162
    plot_arm_proportion().......lines 164-174
variance()......................lines 177-179
run_all_simulations()...........lines 182-259
"""

import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy, scipy.stats


class MAB:
    def __init__(self, a=[2,3,4,5,6], b=[6,5,4,3,2]):
        self.arm2pull = {}
        self.arm2trueExpectedReward = {}
        self.beta_dist(a_vals=a, b_vals=b)
    
    def beta_dist(self, a_vals=[2,3,4,5,6], b_vals=[6,5,4,3,2]):
        """
        Initialize arms according to a beta distribution
        Expected value of beta distribution is a/(a+b)
        Variance is a*b / ((a+b)**2 * (a*b+1)) 
        
        Arguments
        a_vals : list of values for alpha
        b_vals : list of values for beta
        """
        for arm, (a,b) in enumerate(zip(a_vals, b_vals)):
            p = scipy.stats.beta(a, b)
            self.arm2pull[arm] = p
            self.arm2trueExpectedReward[arm] = a/(a+b)
        self.arms = list(self.arm2pull.keys())
        self.mu_star = max(self.arm2trueExpectedReward.values())
    
    def plot_distribution(self):
        """View distributions for each arm."""
        x = np.linspace(0,1,100)
        for arm in self.arms:
            p = self.arm2pull[arm]
            a,b = p.args
            plt.plot(x, p.pdf(x), 
                     label="arm {}: $(a = {}, b = {})$".format(arm,a,b))
        
        ax = plt.gca()
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
        plt.title("Reward distribution per arm")
        plt.xlabel('Reward')
        plt.ylabel('Prob. density')
        plt.tight_layout()
        plt.show();
    
    def play_arm(self, i_t, n=1):
        """
        Get the reward from playing arm i_t, n times.

        Arguments
        i_t : The index of the arm to play
        n : Number of times to play arm i
        """
        return self.arm2pull[i_t].rvs(n)
    
    def gamble(self, T=1000, eps=0.1, m=1, UCB=False):
        """
        General greedy algorithm: 
        Start by playing each arm m times.
        Then at each timestep play a random
        arm with probability epsilon, or the 
        best arm with probability 1-epsilon.

        Greedy algorithm: eps = 0, m=1
        Epsilon greedy: eps > 0, m=1
        Epsilon-first greedy: eps = 0, m>=1
        Random gamble: eps=1, m=1
        """
        self.T = T
        self.rewards = []
        arm2sum = {a:0 for a in self.arms}
        arm2num = {a:0 for a in self.arms}
        # Play each arm once in the beginning
        for t in self.arms:
            i_t = t
            r_t = self.play_arm(i_t, m)
            self.rewards.append(r_t)
            arm2sum[i_t] += r_t.sum()
            arm2num[i_t] += m
        # For the remainder play the arm according to heuristic
        for t in range(len(self.arms)*m, T):
            if UCB:
                arm2mu_est = {a:arm2sum[a]/arm2num[a] for a in self.arms if arm2num[a]>0}
                arm2ucb = {a:(r + np.sqrt((2*np.log(t)) / arm2num[a])) for a,r in arm2mu_est.items()}
                best = max(arm2ucb.values())
                best_arms = [a for a in self.arms if arm2ucb[a] == best]
                i_t = random.choice(best_arms)
            elif random.random() < eps:
                i_t = random.choice(self.arms)
            else:
                arm2mu_est = {a:arm2sum[a]/arm2num[a] for a in self.arms if arm2num[a]>0}
                if arm2mu_est:
                    best = max(arm2mu_est.values())
                    best_arms = [a for a in self.arms if arm2mu_est[a] == best]
                    i_t = random.choice(best_arms)
                else:
                   i_t = random.choice(self.arms)
            r_t = self.play_arm(i_t)
            self.rewards.append(r_t)
            arm2sum[i_t] += r_t
            arm2num[i_t] += 1
        self.rewards = list(itertools.chain(*self.rewards))
        self.arm2num = arm2num
        return T*self.mu_star - sum(self.rewards)
    
    def plot_individual_gamble(self, plt_label):
        """Plot the regret over time for a single run."""
        cum_rewards = np.array(self.rewards).cumsum()
        self.regrets = [t*self.mu_star - cum_rewards[t] for t in range(self.T)]
        plt.plot(range(self.T), self.regrets, label=plt_label)
        plt.legend()
        plt.title("Regret from a single gamble".title())
        plt.xlabel("Duration of gamble $T$")
        plt.ylabel("Regret $R$")
    
    def plot_average_gamble(self, plt_label, eps=0, m=1, num_runs=100, ucb=False):
        """
        Plot the regret over time averaged over a specified number of runs.
        
        Arguments
        plt_label : label for the plot (i.e. random, greedy, epsilon-first...)
        eps : epsilon value to be used in gamble function
        m : number of times to pull each lever before moving on
        num_runs : number of runs to average over
        ucb : (Default False) if true runs the UCB1 algorithm
        """
        self.avg_regret = np.zeros((1, self.T))
        self.total_arm_pulls = {a:0 for a in self.arms}
        for run in range(num_runs):
            self.gamble(self.T, eps=eps, m=m, UCB=ucb)
            cum_rewards = np.array(self.rewards).cumsum()
            for arm in self.arm2num:
                self.total_arm_pulls[arm] += self.arm2num[arm]
            regrets = [t*self.mu_star - cum_rewards[t] for t in range(self.T)]
            self.avg_regret = np.vstack((self.avg_regret, regrets))
        self.avg_regret = np.delete(self.avg_regret, 0, 0)
        self.avg_regret = np.mean(self.avg_regret, axis=0)
        plt.plot(range(self.T), self.avg_regret, label=plt_label)
        plt.title("Average gamble over {} runs".format(num_runs).title())
        plt.xlabel("Duration of gamble $T$")
        plt.ylabel("Regret $R$")
    
    def plot_arm_proportion(self, plt_label):
        """Plot the proportion of time spent playing each arm."""
        T = sum([a for a in self.total_arm_pulls.values()])
        arm2prop_pulls = {a:p/T for a,p in self.total_arm_pulls.items()}
        print(f"{plt_label}: {arm2prop_pulls}")
        plt.bar(x = arm2prop_pulls.keys(),
                height = arm2prop_pulls.values(),
                label = plt_label)
        plt.title("Distribution of pulls for {}".format(plt_label))
        plt.xlabel("Arm")
        plt.ylabel("Proportion of Pulls")


def variance(a,b):
    """Calculate the variance of the Beta distribution."""
    return (a*b) / ((a+b)**2 * (a*b+1)) 


def run_all_simulations(a,b):
    """Run all simulations for random, greedy, epsilon, and epsilon-first."""
    # Initialize seed for replication
    random.seed(1990)
    np.random.seed(2022)
    m=20

    # Prepare separate models for each algorithm
    random_gamble = MAB(a,b)
    greedy_gamble = MAB(a,b)
    epsilon_greedy = MAB(a,b)
    epsilon_first_greedy = MAB(a,b)
    ucb = MAB(a,b)

    # Visualize the reward distribution
    random_gamble.plot_distribution()
    for arm, (a,b) in enumerate(zip(a,b)):
        print(f"Arm {arm}: Expected value: {a/(a+b)}, Variance: {variance(a,b)}")

    # Initialize each algorithm
    random_gamble.gamble(1000, eps=1, m=1)
    greedy_gamble.gamble(1000, eps=0, m=1)
    epsilon_greedy.gamble(1000, eps=0.1, m=1)
    epsilon_first_greedy.gamble(1000, eps=0, m=m)
    ucb.gamble(1000, eps=0, m=1, UCB=True)

    # Compare each algorithm
    random_gamble.plot_individual_gamble("Random Gamble")
    greedy_gamble.plot_individual_gamble("Greedy Gamble")
    epsilon_greedy.plot_individual_gamble("Epsilon-Greedy")
    epsilon_first_greedy.plot_individual_gamble("Epsilon-First")
    plt.title("Regret of an individual gamble")
    plt.tight_layout()
    plt.show();
    
    # Please wait
    print("Plotting the average regret over 100 runs for all algorithms...")

    # Compare average regret
    random_gamble.plot_average_gamble("Random Gamble", eps=1, m=1)
    greedy_gamble.plot_average_gamble("Greedy Gamble", eps=0, m=1)
    epsilon_greedy.plot_average_gamble("Epsilon-Greedy", eps=0.1, m=1)
    epsilon_first_greedy.plot_average_gamble("Epsilon-First", eps=0, m=m)
    plt.title("Average regret over 100 gambles")
    plt.tight_layout()
    plt.legend()
    plt.show();

    # Analyze proportion of lever pulls
    plt.figure(figsize=(10,8))
    plt.subplot(2,2,1)
    random_gamble.plot_arm_proportion("Random Gamble")
    plt.subplot(2,2,2)
    greedy_gamble.plot_arm_proportion("Greedy Gamble")
    plt.subplot(2,2,3)
    epsilon_greedy.plot_arm_proportion("Epsilon-Greedy Gamble")
    plt.subplot(2,2,4)
    epsilon_first_greedy.plot_arm_proportion("Epsilon-First")
    # plt.subplot(3,2,5)
    plt.tight_layout()
    plt.show();

    # Compare UCB to epsilon first
    print("Now comparing epsilon-first to UCB")
    ucb.plot_average_gamble("UCB1", eps=0, m=1, ucb=True)
    epsilon_first_greedy.plot_average_gamble("Epsilon-First", eps=0, m=m)
    plt.title("Regret of an individual gamble")
    plt.title("Average regret over 100 gambles")
    plt.tight_layout()
    plt.legend()
    plt.show();

    plt.subplot(1,2,1)
    epsilon_first_greedy.plot_arm_proportion("Epsilon-First")
    plt.subplot(1,2,2)
    ucb.plot_arm_proportion("UCB1")
    plt.tight_layout()
    plt.show();

if __name__ == '__main__':
    # Easy
    a = [1,2,0.5,2,9]
    b = [3,7,3,4,1]
    run_all_simulations(a, b)

    # Hard
    a = [0.5, 0.5, 0.5, 0.5, 0.5]
    b = [0.5, 0.55, 0.45, 0.6, .4]
    run_all_simulations(a, b) 
