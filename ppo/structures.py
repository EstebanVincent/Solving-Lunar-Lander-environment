import numpy as np
import torch
from torch.utils.data import Dataset

class Episode:
    def __init__(self, gamma=0.99, lambd=0.95):
        self.states = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_target = []
        self.values = []
        self.log_probabilities = []
        self.gamma = gamma
        self.lambd = lambd

    def append(
        self, state, action, reward, value, log_probability
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probabilities.append(log_probability)

    def end_episode(self, last_value):
        rewards = np.array(self.rewards + [last_value])
        values = np.array(self.values + [last_value])

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantages = cumulative_sum(deltas.tolist(), gamma=self.gamma * self.lambd)
        self.rewards_target = cumulative_sum(rewards.tolist(), gamma=self.gamma)[:-1]



class Epoch(Dataset):
    def __init__(self):
        self.episodes = []
        self.states = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_target = []
        self.log_probabilities = []

    def add_episode(self, episode):
        self.episodes.append(episode)

    def build_dataset(self):
        for episode in self.episodes:
            self.states += episode.states
            self.actions += episode.actions
            self.advantages += episode.advantages
            self.rewards += episode.rewards
            self.rewards_target += episode.rewards_target
            self.log_probabilities += episode.log_probabilities

        assert (
            len(
                {
                    len(self.states),
                    len(self.actions),
                    len(self.advantages),
                    len(self.rewards),
                    len(self.rewards_target),
                    len(self.log_probabilities),
                }
            )
            == 1
        )

        self.advantages = normalize_list(self.advantages)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.advantages[idx],
            self.log_probabilities[idx],
            self.rewards_target[idx],
        )



class FreeFallEpisode:
    def __init__(self, gravity, wind_power, turbulance_power):
        self.gravity = gravity
        self.wind_power = wind_power
        self.turbulance_power = turbulance_power
        self.observation = []

    def append(self, state):
        self.observation += state.tolist()

class FreeFallEpoch:
    def __init__(self, gravity, wind_power, turbulance_power):
        self.gravity = gravity
        self.wind_power = wind_power
        self.turbulance_power = turbulance_power
        self.observations = []

    def add_episode(self, episode):
        self.observations.append(episode.observation)

    def get_dynamics(self):
        dynamics = np.array([self.gravity, self.wind_power, self.turbulance_power])
        return dynamics
    
        


def cumulative_sum(array, gamma=1.0):
    curr = 0
    cumulative_array = []

    for a in array[::-1]:
        curr = a + gamma * curr
        cumulative_array.append(curr)

    return cumulative_array[::-1]

def normalize_list(array):
    array = np.array(array)
    array = (array - np.mean(array)) / (np.std(array) + 1e-5)
    return array.tolist()