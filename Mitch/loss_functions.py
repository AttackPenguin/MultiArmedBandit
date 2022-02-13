import numpy as np


def loss_func(payout, action, arm2reward, arm2r_hat):
    """
    Compare the payout with the mean from the best arm. The standard deviation
    of the arm corresponding to the action is added in an attempt to denoise the 
    data.

    Arguments:
    payout : The reward received from playing an arm
    action : The arm that was played
    arm2reward : Dictionary storing the history of payouts for each arm
    arm2r_hat : Dictionary storing the expected return for each arm
    """
    for arm in arm2reward:
        if arm2reward[arm]:
            r_hat = np.mean(arm2reward[arm])
        else:
            r_hat = 1
        arm2r_hat[arm] = r_hat
    max_r_hat = max(arm2r_hat.values())
    std = np.std(arm2reward[action])
    if payout - max_r_hat != 0:
        reward = float((payout - max_r_hat + std)**-1)
    else:
        reward = 1000
    return reward


def loss_func_mean(payout, rewards, memory=3):
    """
    Compare the current payout with the history from the last n plays.
        
    Arguments
    payout : The reward received from playing an arm
    rewards : History of all the payouts received
    memory : How far back to average the results when making the comparison
    """
    if rewards[-memory:]:
        reward = float((payout - np.mean(np.concatenate(rewards[-memory:])))**-1)
    elif rewards:
        for i in list(range(memory)[::-1]):
            try:
                reward = float((payout - np.mean(np.concatenate(rewards[-i])))**-1)
                break
            except IndexError:
                reward = float((payout - rewards[-1])**-1)
    else:
        reward = float(payout)
    return reward
    

def loss_func_simple(payout, rewards):
    """
    Take the difference between the current payout and the reward received 
    in the last round.

    Arguments
    payout : The reward received from playing an arm
    rewards : History of all the payouts received
    """
    if rewards:
        reward = float((payout - rewards[-1])**-1)
    else:
        reward = float(payout)
    return reward


def loss_func_percent(payout, rewards):
    """
    Take the percent difference between the current 
    payout and the reward received in the last round.

    Arguments
    payout : The reward received from playing an arm
    rewards : History of all the payouts received
    """
    if rewards:
        reward = float(((payout - rewards[-1])/rewards[-1])**-1)
    else:
        reward = float(payout)
    return reward
