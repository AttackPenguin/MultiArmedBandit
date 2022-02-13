import random
import gym
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from loss_functions import loss_func


class MAB(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, a_vals=[2,3,4,5,6], b_vals=[6,5,4,3,2]):
        """
        Initialize distribution for each arm.
        
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

    def step(self, action):
        """
        At each step select an arm to pull, calculate the return value from 
        the arm and feed it into the loss function to get the reward.
        """
        # Record the observation from the action
        self.list_actions.append(action)
        obs = np.array([action])
        payout = self.play_arm(action)
        self.arm2reward[action].append(payout)
        
        # Calculate reward based off of custom loss function
        reward = loss_func(payout, action, self.arm2reward, self.arm2r_hat)
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
        """
        Generate visualization of 
        (top) the distribution of rewards from each lever pull compared
        to the distribution of the arm with the highest expected value. 
        
        (bottom) The regret as a function of time.
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

    def beta_dist(self):
        """
        Initialize arms according to a beta distribution

        Expected value : a/(a+b)
        Variance : a*b / ((a+b)**2 * (a*b+1)) 
        """
        for arm, (a,b) in enumerate(zip(self.a, self.b)):
            p = scipy.stats.beta(a, b)
            self.arm2pull[arm] = p
            self.arm2trueExpectedReward[arm] = a/(a+b)
        self.arms = list(self.arm2pull.keys())
        self.r_star = max(self.arm2trueExpectedReward.values())
        self.best_arm = [i for i in self.arm2trueExpectedReward.keys() if self.arm2trueExpectedReward[i] == self.r_star]

    def play_arm(self, arm, n_pulls=1):
        """Return random samples from playing an arm."""
        return self.arm2pull[arm].rvs(n_pulls)


if __name__ == '__main__':
    # Check environment for compliance with gym
    env = MAB()
    check_env(env, warn=True)

    # Initialize and train the model
    TRAIN_LEN = 1000
    env = MAB()
    env = make_vec_env(lambda: env, n_envs=1)
    model = A2C("MlpPolicy", env, verbose=1).learn(TRAIN_LEN)
    env.render()

    # Run the trained model on 1000 timesteps
    # obs = env.reset()
    # n_steps = 500
    # for step in range(n_steps):
    #     action, _ = model.predict(obs, deterministic=False)
    #     if step < 5 or step > n_steps-5:
    #         print("Step {}".format(step + 1))
    #         print("Action: ", action)
    #         obs, reward, done, info = env.step(action)
    #         print('obs=', obs, 'reward=', reward, 'done=', done, 'payout=', info)
    #     else:
    #         obs, regret, done, info = env.step(action)
    # env.render()

    # Good explanation why we set deterministic to False
    # https://stackoverflow.com/questions/66455636/what-does-deterministic-true-in-stable-baselines3-library-means