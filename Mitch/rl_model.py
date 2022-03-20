import os
import random
import gym
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from collections import Counter
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from loss_functions import loss_func
# from callbacks import SaveOnBestTrainingRewardCallback


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

        return self.list_actions

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

def train_model(training_len = 1000, g=0.95, lr=0.0007, steps=1):
    """Train the model."""
    env = MAB()
    # env = make_vec_env(lambda: env, n_envs=1)
    model = A2C("MlpPolicy",
                env, verbose=1, 
                gamma=g,
                learning_rate=lr,
                n_steps=steps).learn(training_len)
    env.render()
    return model

def eval_model(env, model):
    """Evaluate model on new data"""
    env.reset()
    model.predict()

def regret(a_val,b_val,actions,T):
    expected_value = []
    for a,b in zip(a_val, b_val):
        expected_value.append(a/(a+b))
    r_star = max(expected_value)
    regret_list = [t*r_star - cums]
    return T * r_star - 
        

if __name__ == '__main__':
    # Check environment for compliance with gym
    env = MAB()
    check_env(env, warn=True)

    T = 100
    # Easy
    a = [1,1,1,1,20]
    b = [2,3,4,5,1]

    # Hard 
    # a = [1,1,1,1,1]
    # b = [2,2.5,1.5,1.75,2.25]
    # Create and wrap the environment
    env = MAB(a_vals=a, b_vals=b)
    model = A2C('MlpPolicy', 
                env, gamma=0.1,
                learning_rate=0.0006,
                n_steps=1,
                vf_coef=0.9,
                verbose=1).learn(T)
    train_arm_list = env.render()
    train_arm_list
    arm2counts = Counter(train_arm_list)
    plt.bar(arm2counts.keys(), arm2counts.values())
    plt.show();

    # Test the trained model
    test_arm_list = []
    obs = env.reset()
    for step in range(1000-T):
        action, _ = model.predict(obs, deterministic=False)
        test_arm_list.append(action)
    
    action_list = train_arm_list + test_arm_list

    # 
    # model = train_model(training_len=100, g=0.1, lr=0.0006)

    # model.save('mods/A2C_g88_lr0008')
    # model = A2C.load('mods/A2C_g88_lr0008')