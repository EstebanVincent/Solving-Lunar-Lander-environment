import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm

from ppo.structures import FreeFallEpisode
from ppo.networks import ActorNetwork,  DynamicsIdNetwork
from ppo.utils import d_state, random_dynamics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Demo:
    def __init__(self, render):
        self.model_dir = "model/actor_critic/demo"
        self.render = render

        self.state_dim = 8
        self.action_dim = 4
        self.lr = 1e-3

        self.actor_network = ActorNetwork(
            self.state_dim, self.action_dim, torch.optim.Adam, self.lr)
        self.actor_network.load_state_dict(torch.load(
            f"{self.model_dir}/actor_model.pkl"))

    def evaluate(self, n_episodes=10):
        print(f"{'-'*25}Evaluation started{'-'*25}")
        env = gym.make("LunarLander-v2")
        if self.render:
            env = gym.make('LunarLander-v2', render_mode='human')

        scores = []
        for episode in tqdm(range(n_episodes), desc='Evaluate'):
            state, _ = env.reset()
            score = 0
            while True:
                state = torch.from_numpy(state).float().to(device)
                action, _ = self.actor_network.select_action(state)
                next_state, reward, done, *_ = env.step(action)
                score += reward
                if done:
                    break
                state = next_state
            scores.append(score)
        env.close()
        print(f"{'-'*25}Evaluation finished{'-'*25}")
        print('Mean Score:', np.mean(scores))


class DiscoverEval:
    def __init__(self, d_version):
        self.model_dir = "model"
        self.d_version = d_version

        self.state_dim = 8
        self.dynamics_dim = 3
        self.lr = 1e-3

        self.dynamic_id_network = DynamicsIdNetwork(
            self.state_dim*10, self.dynamics_dim, torch.optim.Adam, self.lr).to(device)
        self.dynamic_id_network.load_state_dict(torch.load(
            f"{self.model_dir}/dynamic_id/model_v{self.d_version}.pkl"))

    def evaluate(self, n_episodes=10):
        print(f"{'-'*35}Evaluation started{'-'*35}")
        loss_fn = torch.nn.MSELoss()
        losses = []
        for episode in range(n_episodes):
            # generate random dynamic parameters
            gravity, wind_power, turbulance_power = random_dynamics()

            env = gym.make(
                "LunarLander-v2",
                gravity=gravity,
                enable_wind=True,
                wind_power=wind_power,
                turbulence_power=turbulance_power,
            )
            state, _ = env.reset()

            freefall = FreeFallEpisode(gravity, wind_power, turbulance_power)
            freefall.append(state)

            for step in range(9):
                state, *_ = env.step(0)       # do nothing
                freefall.append(state)
            freefall_obs = torch.from_numpy(
                np.array(freefall.observation)).float()

            y_pred = self.dynamic_id_network(freefall_obs)
            y_true = torch.from_numpy(
                np.array([abs(gravity), wind_power, turbulance_power]))
            loss = loss_fn(y_pred, y_true)
            losses.append(loss.item())
            print(
                f"Loss : {loss} | y_pred = {y_pred.detach().numpy()} | y_true = {y_true.numpy()}")
            env.close()
        print(f"{'-'*35}Evaluation finished{'-'*35}")


class Eval:
    def __init__(self, d_version, m_version):
        self.model_dir = "model"
        self.d_version = d_version
        self.m_version = m_version

        self.state_dim = 8
        self.action_dim = 4
        self.dynamics_dim = 3
        self.lr = 1e-3
        self.n_steps = 500

        self.actor_network = ActorNetwork(
            self.state_dim + self.dynamics_dim, self.action_dim, torch.optim.Adam, self.lr).to(device)
        self.actor_network.load_state_dict(torch.load(
            f"{self.model_dir}/actor_critic/actor_model_v{self.m_version}.pkl"))

        self.dynamic_id_network = DynamicsIdNetwork(
            self.state_dim*10, self.dynamics_dim, torch.optim.Adam, self.lr).to(device)
        self.dynamic_id_network.load_state_dict(torch.load(
            f"{self.model_dir}/dynamic_id/model_v{self.d_version}.pkl"))

    def evaluate(self, n_episodes=10, render=False):
        print(f"{'-'*25}Evaluation started{'-'*25}")

        scores = []
        for episode in tqdm(range(n_episodes), desc='Evaluate'):
            # generate random dynamic parameters
            gravity, wind_power, turbulance_power = random_dynamics()
            if not render:
                env = gym.make(
                    "LunarLander-v2",
                    gravity=gravity,
                    enable_wind=True,
                    wind_power=wind_power,
                    turbulence_power=turbulance_power,
                )
            else:
                env = gym.make(
                    "LunarLander-v2",
                    gravity=gravity,
                    enable_wind=True,
                    wind_power=wind_power,
                    turbulence_power=turbulance_power,
                    render_mode='human'
                )

            state, _ = env.reset()

            freefall = FreeFallEpisode(gravity, wind_power, turbulance_power)
            freefall.append(state)

            score = 0
            for step in range(9):
                state, reward, *_ = env.step(0)       # do nothing
                score += reward
                freefall.append(state)
            freefall_obs = torch.from_numpy(
                np.array(freefall.observation)).float()
            id_dynamic_values = self.dynamic_id_network(
                freefall_obs)

            for step in range(9, self.n_steps):
                d_obs = d_state(state, id_dynamic_values)
                action, _ = self.actor_network.select_action(d_obs)
                next_state, reward, done, *_ = env.step(action)
                score += reward
                if done:
                    break
                state = next_state
            scores.append(score)
            env.close()
        print(f"{'-'*25}Evaluation finished{'-'*25}")
        print('Mean Score:', np.mean(scores))
