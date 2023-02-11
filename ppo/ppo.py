import gym
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from ppo.structures import Episode, Epoch, FreeFallEpisode, FreeFallEpoch
from ppo.networks import ActorNetwork, CriticNetwork, DynamicsIdNetwork, device


class Agent:
    def __init__(self, env, version, existing_model=False):
        self.log_dir = "logs/agent"
        self.model_dir = "model"

        self.env = env
        self.version = version

        self.state_dim = self.env.observation_space.shape[0]  # 8
        self.action_dim = self.env.action_space.n  # 4
        self.dynamics_dim = 3
        self.lr = 1e-3

        self.actor_network = ActorNetwork(
            self.state_dim, self.action_dim, torch.optim.Adam, self.lr).to(device)
        self.dynamic_id_network = DynamicsIdNetwork(
            self.state_dim*10, self.dynamics_dim, torch.optim.Adam, self.lr).to(device)

        if not existing_model:

            self.episode_ite = 0
            self.epoch_ite = 0

            self.n_epochs = 100
            self.n_episodes = 20
            self.n_steps = 500

            self.writer = SummaryWriter(log_dir=self.log_dir)  # logs

            self.critic_network = CriticNetwork(
                self.state_dim, torch.optim.Adam, self.lr).to(device)

        else:
            self.actor_network.load_state_dict(torch.load(
                f"{self.model_dir}/actor_model_v{self.version}.pkl")).to(device)
            self.dynamic_id_network.load_state_dict(torch.load(
                f"{self.model_dir}/dynamic_id_model_v{self.version}.pkl")).to(device)

    def train(self):
        print(f"{'-'*25}Training started{'-'*25}")
        for i_epoch in tqdm(range(self.n_epochs), desc="Epochs"):
            epoch = Epoch()
            for i_episode in range(self.n_episodes):
                state, _ = self.env.reset()
                episode = Episode()
                for step in range(self.n_steps):
                    state = torch.from_numpy(state).float().to(device)

                    action, log_prob = self.actor_network.select_action(state)
                    value = self.critic_network.predict(state)
                    next_state, reward, done, *_ = self.env.step(action)

                    episode.append(state, action, reward, value, log_prob)
                    state = next_state
                    if done:
                        episode.end_episode(last_value=0)
                        break
                    if step == self.n_steps - 1:
                        value = self.critic_network.predict(state)
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
                    "Episode Lenght",
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
        print(f"{'-'*25}Training Finished{'-'*25}")
        torch.save(self.actor_network.state_dict(),
                   f"{self.model_dir}/actor_model_v{self.version}.pkl")
        torch.save(self.critic_network.state_dict(),
                   f"{self.model_dir}/critic_model_v{self.version}.pkl")

    def evaluate(self, n_episodes=10):
        print(f"{'-'*25}Evaluation started{'-'*25}")

        scores = []
        for episode in tqdm(range(n_episodes), desc='Evaluate'):
            state, _ = self.env.reset()
            score = 0
            while True:
                state = torch.from_numpy(state).float().to(device)
                action, _ = self.actor_network.select_action(state)
                next_state, reward, done, *_ = self.env.step(action)
                score += reward
                if done:
                    break
                state = next_state
            scores.append(score)
        print(f"{'-'*25}Evaluation finished{'-'*25}")
        print('Mean Score:', np.mean(scores))


class Analysis:
    def __init__(self, version):
        self.log_dir = "logs/analysis"
        self.model_dir = "model"

        self.version = version

        self.state_dim = 8
        self.dynamics_dim = 3
        self.lr = 1e-3

        self.episode_ite = 0
        self.epoch_ite = 0

        self.n_epochs = 100
        self.n_episodes = 20
        self.n_steps = 10

        self.dynamic_id_network = DynamicsIdNetwork(
            self.state_dim*10, self.dynamics_dim, torch.optim.Adam, self.lr).to(device)
        self.writer = SummaryWriter(log_dir=self.log_dir)  # logs

    def dynamics_identification(self, env, gravity, wind_power, turbulance_power):
        freefall = FreeFallEpisode(gravity, wind_power, turbulance_power)
        starting_state, _ = env.reset()
        freefall.append(starting_state)
        for step in range(9):  # counting starting step
            next_state, *_ = env.step(0)       # do nothing
            freefall.append(next_state)
        loss = self.dynamic_id_network.train(freefall)
        freefall.loss = loss
        return loss

    def train(self):
        print(f"{'-'*25}Training started{'-'*25}")
        for i_epoch in tqdm(range(self.n_epochs), desc="Epochs"):
            gravity, wind_power, turbulance_power = self.random_dynamics()
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
        print(f"{'-'*25}Training Finished{'-'*25}")
        torch.save(self.dynamic_id_network.state_dict(),
                   f"{self.model_dir}/dynamic_id_model_v{self.version}.pkl")

    @staticmethod
    def random_dynamics():
        gravity = np.round(np.random.uniform(-11.9, -0.1), 1)
        wind_power = np.round(np.random.uniform(0.1, 20), 1)
        turbulance_power = np.round(np.random.uniform(0.1, 2), 1)
        return gravity, wind_power, turbulance_power
