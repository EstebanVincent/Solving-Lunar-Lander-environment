import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_dynamics():
    gravity = np.random.randint(-11, 0)
    wind_power = np.random.randint(1, 20)
    turbulance_power = np.round(np.random.uniform(0.1, 2), 1)
    return gravity, wind_power, turbulance_power


def normalize_list(array):
    array = np.array(array)
    array = (array - np.mean(array)) / (np.std(array) + 1e-5)
    return array.tolist()


def cumulative_sum(array, gamma=1.0):
    """
    The discount factor is used to balance the trade-off between immediate and future rewards.
    """
    curr = 0
    cumulative_array = []

    for a in array[::-1]:
        curr = a + gamma * curr
        cumulative_array.append(curr)

    return cumulative_array[::-1]


def d_state(state, d_args):
    state = torch.from_numpy(np.array(state)).float()
    d_obs = torch.cat((state, d_args), dim=0)
    return d_obs


def d_epoch_state(states, d_args):
    states = torch.from_numpy(np.array(states)).float()
    d_args = d_args.unsqueeze(0)
    d_obs = torch.cat((states, d_args.repeat(len(states), 1)), dim=1)
    return d_obs
