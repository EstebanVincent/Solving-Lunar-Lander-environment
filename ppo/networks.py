import torch
from torch.distributions import Categorical
from torch.nn import Module, Linear, LeakyReLU
import numpy as np


class ActorNetwork(Module):
    def __init__(self, input_size, output_size, optimizer, lr):
        super(ActorNetwork, self).__init__()
        self.output_size = output_size
        self.fc1 = Linear(input_size, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, 64)
        self.fc4 = Linear(64, output_size)

        self.l_relu = LeakyReLU(0.1)
        self.optimizer = optimizer(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))

        y = torch.softmax(self.fc4(x), dim=-1)

        return y

    def select_action(self, d_state):
        y = self(d_state)
        prediction = Categorical(y)
        action = prediction.sample()

        log_probability = prediction.log_prob(action)

        return action.item(), log_probability.item()

    def evaluate_actions(self, d_states, actions):
        y = self(d_states)

        dist = Categorical(y)

        entropy = dist.entropy()

        log_probabilities = dist.log_prob(actions)

        return log_probabilities, entropy

    def train(self, data_loader, epochs=4, clip=0.2):
        epochs_losses = []
        c1 = 0.01

        for i in range(epochs):
            losses = []
            for d_states, actions, advantages, log_probabilities, _ in data_loader:
                self.optimizer.zero_grad()

                new_log_probabilities, entropy = self.evaluate_actions(
                    d_states, actions)

                loss = (self.ac_loss(new_log_probabilities, log_probabilities,
                        advantages, clip,).mean() - c1 * entropy.mean())
                loss.backward(retain_graph=True)
                self.optimizer.step()

                losses.append(loss.item())

            mean_loss = np.mean(losses)

            epochs_losses.append(mean_loss)

        return epochs_losses

    @staticmethod
    def ac_loss(new_log_probabilities, old_log_probabilities, advantages, epsilon_clip):
        probability_ratios = torch.exp(
            new_log_probabilities - old_log_probabilities)
        clipped_probabiliy_ratios = torch.clamp(
            probability_ratios, 1 - epsilon_clip, 1 + epsilon_clip
        )

        surrogate_1 = probability_ratios * advantages
        surrogate_2 = clipped_probabiliy_ratios * advantages

        return -torch.min(surrogate_1, surrogate_2)


class CriticNetwork(Module):
    def __init__(self, input_size, optimizer, lr):
        super(CriticNetwork, self).__init__()
        self.fc1 = Linear(input_size, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, 64)
        self.fc4 = Linear(64, 1)

        self.l_relu = LeakyReLU(0.1)
        self.optimizer = optimizer(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))

        y = self.fc4(x)

        return y.squeeze(1)

    def predict(self, d_state):
        if not isinstance(d_state, torch.Tensor):
            d_state = torch.from_numpy(d_state).float()
        if len(d_state.size()) == 1:
            d_state = d_state.unsqueeze(0)

        y = self(d_state)

        return y.item()

    def train(self, data_loader, epochs=4):
        epochs_losses = []
        for i in range(epochs):
            losses = []
            for d_state, _, _, _, rewards_target in data_loader:
                rewards_target = rewards_target.float()

                self.optimizer.zero_grad()

                values = self(d_state)
                
                loss = (values - rewards_target).pow(2).mean()

                loss.backward(retain_graph=True)

                self.optimizer.step()

                losses.append(loss.item())

            mean_loss = np.mean(losses)

            epochs_losses.append(mean_loss)

        return epochs_losses


class DynamicsIdNetwork(Module):
    """
    predict the opposite gravity, wind power, turbulance power
    """
    # input = state_size*10 = 80, output = 3
    def __init__(self, input_size, output_size, optimizer, lr):
        super(DynamicsIdNetwork, self).__init__()
        self.fc1 = Linear(input_size, 32)
        self.fc2 = Linear(32, 8)
        self.fc3 = Linear(8, output_size)

        self.l_relu = LeakyReLU(0.1)
        self.optimizer = optimizer(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))

        x = self.fc3(x)
        return x

    def train(self, epoch, epochs=10):
        epochs_losses = []
        loss_fn = torch.nn.MSELoss()
        for i_epoch in range(epochs):
            losses = []
            for observation in epoch.observations:
                self.optimizer.zero_grad()
                identified_values = self(torch.from_numpy(np.array(observation)).float())
                true_values = torch.from_numpy(epoch.get_dynamics_inv_g()).float()
                loss = loss_fn(identified_values, true_values)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            mean_loss = np.mean(losses)
            epochs_losses.append(mean_loss)

        return epochs_losses