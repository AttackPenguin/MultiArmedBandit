from rl_model import *
from heuristic_mab import *


env = MAB()
check_env(env, warn=True)

T = 100
a = [1,2,0.5,2,9]
b = [3,7,3,4,1]
env = MAB(a_vals=a, b_vals=b)
model = A2C('MlpPolicy',
            env,
            gamma=0.1,
            learning_rate=0.0008,
            n_steps=1,
            vf_coef=0.9,
            verbose=1).learn(T)
train_arm_list = env.render()
train_arm_list
arm2counts = Counter(train_arm_list)
plt.bar(arm2counts.keys(), arm2counts.values())
plt.show();