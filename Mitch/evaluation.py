from rl_model import *
from heuristic_mab import *

if __name__ == "__main__":
    # Check environment for compliance with gym
    env = MAB()
    check_env(env, warn=True)

    # Initialize timestep and beta dist. parameters
    T = 1000
    # Easy
    a = [1,2,0.5,2,9]
    b = [3,7,3,4,1]

    # Hard
    # a = [0.5, 0.5, 0.5, 0.5, 0.5]
    # b = [0.5, 0.55, 0.45, 0.6, .4]

    # Initialize environment and model
    env = MAB(a_vals=a, b_vals=b)
    model = A2C('MlpPolicy',
                env,
                gamma=0.1,
                learning_rate=0.0008,
                n_steps=1,
                vf_coef=0.9,
                verbose=1).learn(T)
    train_arm_list = env.render()

    # print(train_arm_list)
    model_name = "TEST_1"
    # model.save(f'mods/{model_name}')
    # model = A2C.load(f"mods/{model_name}")

    # Plot distribution of pulls
    # arm2counts = Counter(train_arm_list)
    # plt.title("Distribution of Pulls")
    # plt.xlabel("Arm")
    # plt.ylabel("Number of pulls")
    # plt.bar(arm2counts.keys(), arm2counts.values())
    # plt.savefig("distribution_of_pulls.png")
    # plt.close()

    # Test the trained model
    # test_arm_list = []
    # obs = env.reset()
    # for step in range(1000-T):
    #     action, _ = model.predict(obs, deterministic=False)
    #     test_arm_list.append(action)
    
    # action_list = train_arm_list + test_arm_list
