import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from ppo.structures import Episode, Epoch, FreeFallEpisode, FreeFallEpoch
from ppo.networks import ActorNetwork, CriticNetwork, DynamicsIdNetwork
from ppo.utils import d_state, random_dynamics, device


class DemoTrain:
    def __init__(self, m_version, n_epochs):
        self.log_dir = "logs/actor_critic/demo"
        self.model_dir = "model/actor_critic/demo"
        env = gym.make("LunarLander-v2",)
        self.env = env
        self.fake_random = torch.from_numpy(np.array([10, 0, 0]))
        self.m_version = m_version

        self.state_dim = self.env.observation_space.shape[0]  # 8
        self.action_dim = self.env.action_space.n  # 4
        self.random_dim = 3
        self.dynamics_dim = 3
        self.lr = 1e-3

        self.actor_network = ActorNetwork(
            self.state_dim + self.random_dim, self.action_dim, torch.optim.Adam, self.lr).to(device)
        self.critic_network = CriticNetwork(
            self.state_dim + self.random_dim, torch.optim.Adam, self.lr).to(device)

        self.episode_ite = 0
        self.epoch_ite = 0

        self.n_epochs = n_epochs
        self.n_episodes = 20
        self.n_steps = 500

        self.writer = SummaryWriter(log_dir=self.log_dir)  # logs

    def train(self):
        print(f"{'-'*25}Training started{'-'*25}")
        for i_epoch in tqdm(range(self.n_epochs), desc="Epochs"):
            epoch = Epoch()
            for i_episode in range(self.n_episodes):
                state, _ = self.env.reset()
                episode = Episode(self.fake_random)
                for step in range(self.n_steps):

                    action, log_prob = self.actor_network.select_action(
                        d_state(state, self.fake_random))
                    value = self.critic_network.predict(
                        d_state(state, self.fake_random))
                    next_state, reward, done, *_ = self.env.step(action)

                    episode.append(state, action, reward, value, log_prob)
                    state = next_state
                    if done:
                        episode.end_episode(last_value=0)
                        break
                    if step == self.n_steps - 1:
                        value = self.critic_network.predict(
                            d_state(state, self.fake_random))
                        episode.end_episode(last_value=value)

                self.episode_ite += 1
                self.writer.add_scalar(
                    "Episode Score",
                    np.sum(episode.rewards),
                    self.episode_ite,
                )
                self.writer.add_scalar(
                    "Probabilities",
                    np.exp(np.mean(episode.log_probabilities)),
                    self.episode_ite,
                )
                self.writer.add_scalar(
                    "Episode Length",
                    len(episode.rewards),
                    self.episode_ite,
                )
                epoch.add_episode(episode)
            epoch.build_dataset()
            data_loader = DataLoader(epoch, batch_size=32)

            actor_losses = self.actor_network.train(data_loader)
            critic_losses = self.critic_network.train(data_loader)

            for a_loss, c_loss in zip(actor_losses, critic_losses):
                self.epoch_ite += 1
                self.writer.add_scalar("Actor Loss", a_loss, self.epoch_ite)
                self.writer.add_scalar("Critic Loss", c_loss, self.epoch_ite)
        print(f"{'-'*25}Training Finished{'-'*25}")
        torch.save(self.actor_network.state_dict(),
                   f"{self.model_dir}/actor_model_v{self.m_version}.pkl")
        torch.save(self.critic_network.state_dict(),
                   f"{self.model_dir}/critic_model_v{self.m_version}.pkl")


class DiscoverTrain:
    def __init__(self, d_version, n_epochs):
        self.log_dir = "logs/dynamic"
        self.model_dir = "model"

        self.d_version = d_version

        self.state_dim = 8
        self.dynamics_dim = 3
        self.lr = 1e-3

        self.episode_ite = 0
        self.epoch_ite = 0

        self.n_epochs = n_epochs
        self.n_episodes = 20
        self.n_steps = 10

        self.dynamic_id_network = DynamicsIdNetwork(
            self.state_dim*10, self.dynamics_dim, torch.optim.Adam, self.lr).to(device)
        self.writer = SummaryWriter(log_dir=self.log_dir)  # logs

    def train(self):
        print(f"{'-'*25}Training started{'-'*25}")
        for i_epoch in tqdm(range(self.n_epochs), desc="Epochs"):

            gravity, wind_power, turbulance_power = random_dynamics()
            epoch = FreeFallEpoch(gravity, wind_power, turbulance_power)
            env = gym.make(
                "LunarLander-v2",
                gravity=gravity,
                enable_wind=True,
                wind_power=wind_power,
                turbulence_power=turbulance_power,
            )
            for i_episode in range(self.n_episodes):
                episode = FreeFallEpisode(
                    gravity, wind_power, turbulance_power)
                starting_state, _ = env.reset()
                episode.append(starting_state)
                for step in range(self.n_steps - 1):  # counting starting step
                    next_state, *_ = env.step(0)       # do nothing
                    episode.append(next_state)
                self.episode_ite += 1
                epoch.add_episode(episode)
            dynamic_losses = self.dynamic_id_network.train(epoch)

            for d_loss in dynamic_losses:
                self.epoch_ite += 1
                self.writer.add_scalar("Dynamic Loss", d_loss, self.epoch_ite)
            env.close()
            if i_epoch % 50_000 == 0 and i_epoch != 0:
                print(f"\nSaving model at epoch {i_epoch}")
                torch.save(self.dynamic_id_network.state_dict(
                ), f"{self.model_dir}/dynamic_id/model_v{self.d_version}_{i_epoch}.pkl")
        print(f"{'-'*25}Training Finished{'-'*25}")
        torch.save(self.dynamic_id_network.state_dict(),
                   f"{self.model_dir}/dynamic_id/model_v{self.d_version}.pkl")


class Train:
    def __init__(self, d_version, m_version, n_epochs):
        self.log_dir = "logs/actor_critic"
        self.model_dir = "model"

        self.d_version = d_version
        self.m_version = m_version

        self.state_dim = 8
        self.action_dim = 4
        self.dynamics_dim = 3
        self.lr = 1e-3

        self.episode_ite = 0
        self.epoch_ite = 0

        self.n_epochs = n_epochs
        self.n_episodes = 20
        self.n_steps = 500

        self.writer = SummaryWriter(log_dir=self.log_dir)  # logs

        # Initialise the networks
        self.dynamic_id_network = DynamicsIdNetwork(
            self.state_dim*10, self.dynamics_dim, torch.optim.Adam, self.lr).to(device)
        self.actor_network = ActorNetwork(
            self.state_dim + self.dynamics_dim, self.action_dim, torch.optim.Adam, self.lr).to(device)
        self.critic_network = CriticNetwork(
            self.state_dim + self.dynamics_dim, torch.optim.Adam, self.lr).to(device)

        # Load the identification network d_version
        self.dynamic_id_network.load_state_dict(torch.load(
            f"{self.model_dir}/dynamic_id/model_v{self.d_version}.pkl"))

    def train(self):
        print(f"{'-'*25}Training started{'-'*25}")
        for i_epoch in tqdm(range(self.n_epochs), desc="Epochs"):
            gravity, wind_power, turbulance_power = random_dynamics()
            env = gym.make(
                "LunarLander-v2",
                gravity=gravity,
                enable_wind=True,
                wind_power=wind_power,
                turbulence_power=turbulance_power,
            )
            epoch = Epoch()
            for i_episode in range(self.n_episodes):
                freefall = FreeFallEpisode(
                    gravity, wind_power, turbulance_power)

                state, _ = env.reset()
                freefall.append(state)

                for step in range(9):
                    state, *_ = env.step(0)       # do nothing
                    freefall.append(state)
                freefall_obs = torch.from_numpy(
                    np.array(freefall.observation)).float()
                id_dynamic_values = self.dynamic_id_network(
                    freefall_obs)
                episode = Episode(id_dynamic_values)
                for step in range(9, self.n_steps):
                    d_obs = d_state(state, id_dynamic_values)
                    action, log_prob = self.actor_network.select_action(d_obs)
                    value = self.critic_network.predict(d_obs)
                    next_state, reward, done, *_ = env.step(action)

                    episode.append(state, action, reward, value, log_prob)
                    state = next_state
                    if done:
                        episode.end_episode(last_value=0)
                        break
                    if step == self.n_steps - 1:
                        value = self.critic_network.predict(
                            d_state(state, id_dynamic_values))
                        episode.end_episode(last_value=value)

                self.episode_ite += 1
                self.writer.add_scalar(
                    "Episode Score",
                    np.sum(episode.rewards),
                    self.episode_ite,
                )
                self.writer.add_scalar(
                    "Probabilities",
                    np.exp(np.mean(episode.log_probabilities)),
                    self.episode_ite,
                )
                self.writer.add_scalar(
                    "Episode Length",
                    len(episode.rewards),
                    self.episode_ite,
                )
                epoch.add_episode(episode)
            epoch.build_dataset()
            data_loader = DataLoader(epoch, batch_size=32, shuffle=True)

            actor_losses = self.actor_network.train(data_loader)
            critic_losses = self.critic_network.train(data_loader)

            for a_loss, c_loss in zip(actor_losses, critic_losses):
                self.epoch_ite += 1
                self.writer.add_scalar("Actor Loss", a_loss, self.epoch_ite)
                self.writer.add_scalar("Critic Loss", c_loss, self.epoch_ite)
            env.close()
            if i_epoch % 50 == 0 and i_epoch != 0:
                print(f"\nSaving model at epoch {i_epoch}")
                torch.save(self.actor_network.state_dict(
                ), f"{self.model_dir}/actor_critic/actor_model_v{self.m_version}_{i_epoch}.pkl")
                torch.save(self.critic_network.state_dict(
                ), f"{self.model_dir}/actor_critic/critic_model_v{self.m_version}_{i_epoch}.pkl")
        print(f"{'-'*25}Training Finished{'-'*25}")
        torch.save(self.actor_network.state_dict(),
                   f"{self.model_dir}/actor_critic/actor_model_v{self.m_version}.pkl")
        torch.save(self.critic_network.state_dict(),
                   f"{self.model_dir}/actor_critic/critic_model_v{self.m_version}.pkl")
