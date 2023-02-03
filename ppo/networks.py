import torch
from torch.distributions import Categorical
from torch.nn import Module, Linear, LeakyReLU
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    def select_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(device)

        if len(state.size()) == 1:
            state = state.unsqueeze(0)
            
        y = self(state)
        prediction = Categorical(y)
        action = prediction.sample()

        log_probability = prediction.log_prob(action)

        return action.item(), log_probability.item()
    
    def evaluate_actions(self, states, actions):
        y = self(states)

        dist = Categorical(y)

        entropy = dist.entropy()

        log_probabilities = dist.log_prob(actions)

        return log_probabilities, entropy

    def train(self, data_loader, epochs=4, clip=0.2):
        epochs_losses = []
        c1 = 0.01

        for i in range(epochs):
            losses = []
            for observations, actions, advantages, log_probabilities, _ in data_loader:
                observations = observations.float().to(device)
                actions = actions.long().to(device)
                advantages = advantages.float().to(device)
                old_log_probabilities = log_probabilities.float().to(device)

                self.optimizer.zero_grad()

                new_log_probabilities, entropy = self.evaluate_actions(observations, actions)

                loss = (self.ac_loss(new_log_probabilities, old_log_probabilities, advantages, clip,).mean()- c1 * entropy.mean())
                loss.backward()

                self.optimizer.step()

                losses.append(loss.item())

            mean_loss = np.mean(losses)

            epochs_losses.append(mean_loss)

        return epochs_losses

    @staticmethod
    def ac_loss(new_log_probabilities, old_log_probabilities, advantages, epsilon_clip):
        probability_ratios = torch.exp(new_log_probabilities - old_log_probabilities)
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
    
    def predict(self, state):            
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float().to(device)
        if len(state.size()) == 1:
            state = state.unsqueeze(0)

        y = self(state)

        return y.item()
    
    def train(self, data_loader, epochs=4):
        epochs_losses = []
        for i in range(epochs):
            losses = []
            for states, _, _, _, rewards_target in data_loader:
                states = states.float().to(device)
                rewards_target = rewards_target.float().to(device)

                self.optimizer.zero_grad()

                values = self(states)

                loss = (values - rewards_target).pow(2).mean()

                loss.backward()

                self.optimizer.step()

                losses.append(loss.item())

            mean_loss = np.mean(losses)

            epochs_losses.append(mean_loss)

        return epochs_losses