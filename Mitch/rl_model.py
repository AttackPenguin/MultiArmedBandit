import random
import gym
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.env_util import make_vec_env


class MAB(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, a_vals=[2,3,4,5,6], b_vals=[6,5,4,3,2]):
        """Initialize distribution for each arm.
        
        Arguments
        a_vals : alpha values corresponding to a beta distribution
        b_vals : beta values corresponding to a beta distribution
        """
        super(MAB, self).__init__()
        
        # Initialize alpha and beta
        self.a = a_vals
        self.b = b_vals
        assert len(self.a) == len(self.b), "The len of a_vals and b_vals must match."
        self.n_arms = len(self.a)

        # Initialize dictionary to store actions
        self.arm2pull = {}
        self.arm2trueExpectedReward = {}
        self.rewards = []
        self.list_regrets = []
        self.list_actions = []
        self.T = 0

        # Each action comprises of picking one of the arms
        self.action_space = spaces.Discrete(self.n_arms)
        self.observation_space = spaces.Box(low=0, high=self.n_arms, 
                                            shape=(1,), dtype=np.float32)
        
        # Initialize the distribution
        self.beta_dist()
        self.arm2reward = {arm:[] for arm in self.arm2pull}
        self.arm2r_hat = {arm:[] for arm in self.arm2pull}

    def reset(self):
        """Reset the experiment and return a random value."""
        self.rewards = []
        self.list_regrets = []
        self.T = 0
        # return np.array([0]).astype(np.float32)
        arm = random.choice(range(self.n_arms))
        return self.play_arm(arm)
    
    def loss_func(self, payout, action):
        """Compare the payout with the mean of all the arms."""
        for arm in self.arm2reward:
            if self.arm2reward[arm]:
                r_hat = np.mean(self.arm2reward[arm])
            else:
                r_hat = 1
            self.arm2r_hat[arm] = r_hat
        max_r_hat = max(self.arm2r_hat.values())
        std = np.std(self.arm2reward[action])
        if payout - max_r_hat != 0:
            reward = float((payout - max_r_hat + std)**-1)
        else:
            reward = 1000
        return reward

    def loss_func_mean(self, payout, memory=3):
        """Custom loss function. Compare the payout with the history from the last
        n pulls (memory).
        
        Arguments
        payout : reward from playing arm i in the current timestep
        memory : how far back to average the results when making the comparison
        """
        if self.rewards[-memory:]:
            reward = float((payout - np.mean(np.concatenate(self.rewards[-memory:])))**-1)
        elif self.rewards:
            for i in list(range(memory)[::-1]):
                try:
                    reward = float((payout - np.mean(np.concatenate(self.rewards[-i])))**-1)
                    break
                except IndexError:
                    reward = float((payout - self.rewards[-1])**-1)
        else:
            reward = float(payout)
        return reward
    
    def loss_func_simple(self, payout):
        """Custom loss function. Take the difference between the current reward 
        and the reward received in the last round.
        """
        if self.rewards:
            reward = float((payout - self.rewards[-1])**-1)
        else:
            reward = float(payout)
        return reward
    
    def loss_func_percent(self, payout):
        """Custom loss function. Take the percent difference between the current 
        reward and the reward received in the last round.
        """
        if self.rewards:
            reward = float(((payout - self.rewards[-1])/self.rewards[-1])**-1)
        else:
            reward = float(payout)
        return reward

    def step(self, action):
        """Each step select an arm to pull, calculate the return value from 
        the arm and feed it into the loss function to get the reward."""
        # Record the observation from the action
        self.list_actions.append(action)
        obs = np.array([action])
        payout = self.play_arm(action)
        self.arm2reward[action].append(payout)
        
        # Calculate reward based off of custom loss function
        # reward = self.loss_func_mean(payout)
        reward = self.loss_func(payout, action)
        self.rewards.append(payout)

        # Keep track of time for analysis
        info = {self.T:payout[0]}
        self.T += 1

        # Keep track of regret for analysis
        regret = self.T * self.r_star - sum(self.rewards)
        self.list_regrets.append(regret)
        done = False

        return obs, reward, done, info
    
    def render(self, mode='console'):
        """Generate visualization of 
        (top) the distribution of rewards from each
        lever pull compared to the distribution of the arm with the highest 
        expected value. 
        
        (bottom) The regret as time progresses.
        """
        x = np.linspace(0,1,100)

        # Plot distribution of rewards received compared to best distribution
        plt.subplot(2,1,1)
        plt.hist(np.concatenate(self.rewards), bins='auto', density=True)
        plt.plot(x, np.array(self.arm2pull[self.best_arm[0]].pdf(x)))
        plt.ylabel("Frequency")
        plt.xlabel("Reward $R$")

        # Plot regret as a function of time
        plt.subplot(2,1,2)
        plt.plot(np.arange(self.T), self.list_regrets, label="RL algorithm") 
        plt.xlabel("Duration of gamble $T$")
        plt.ylabel("Regret $R$")
        plt.legend()
        plt.tight_layout()
        plt.show();
        
        # Plot regret/T to see if it's zero-strategy
        # plt.subplot(3,1,3)
        # plt.plot(np.arange(self.T), np.array(self.list_regrets) / np.arange(self.T),
        #          label="$R/T$")
        # plt.xlabel("Duration of gamble $T$")
        # plt.ylabel("Expected regret over time $R/T$")

    def beta_dist(self):
        """Initialize arms according to a beta distribution
        Expected value of beta distribution is a/(a+b)
        Variance is a*b / ((a+b)**2 * (a*b+1)) 
        """
        for arm, (a,b) in enumerate(zip(self.a, self.b)):
            p = scipy.stats.beta(a, b)
            self.arm2pull[arm] = p
            self.arm2trueExpectedReward[arm] = a/(a+b)
        self.arms = list(self.arm2pull.keys())
        self.r_star = max(self.arm2trueExpectedReward.values())
        self.best_arm = [i for i in self.arm2trueExpectedReward.keys() if self.arm2trueExpectedReward[i] == self.r_star]

    def play_arm(self, i, n_pulls=1):
        """Return (n_pulls) random samples from arm i."""
        return self.arm2pull[i].rvs(n_pulls)

# Check environment for compliance with gym
# env = MAB()
# check_env(env, warn=True)

################ Using stable baselines #######################
# Initialize and train the model
TRAIN_LEN = 1000
env = MAB()
env = make_vec_env(lambda: env, n_envs=1)
model = A2C("MlpPolicy", env, verbose=1).learn(TRAIN_LEN)
env.render()

# Run the final model on 1000 timesteps
obs = env.reset()
n_steps = 1000
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=False)
    if step < 5 or step > n_steps-5:
        print("Step {}".format(step + 1))
        print("Action: ", action)
        obs, reward, done, info = env.step(action)
        print('obs=', obs, 'reward=', reward, 'done=', done, 'payout=', info)
    else:
        obs, regret, done, info = env.step(action)
env.render()

# Good explanation why we set deterministic to False
# https://stackoverflow.com/questions/66455636/what-does-deterministic-true-in-stable-baselines3-library-means