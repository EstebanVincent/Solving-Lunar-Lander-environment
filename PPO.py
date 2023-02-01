import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x



class PPO:
    def __init__(self, env, input_size, hidden_size, output_size, fname = None, n_epochs=1000, n_episodes=10, n_steps=400, epsilon=0.1, lr=0.001):
        self.env = env
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.policy_net = PolicyNetwork(self.input_size, self.hidden_size, self.output_size)
        if fname is None:
            self.n_epochs = n_epochs
            self.n_episodes = n_episodes
            self.n_steps = n_steps
            self.epsilon = epsilon
            self.lr = lr
            self.old_policy_net = PolicyNetwork(self.input_size, self.hidden_size, self.output_size)
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        else :
            self.policy_net.load_state_dict(torch.load(f"models/{fname}.pkl"))

    def select_action(self, state):
        if not state is torch.Tensor:
            state = torch.from_numpy(state).float().to(device)
        if len(state.size()) == 1:
            state = state.unsqueeze(0)
            
        probs = self.policy_net(state)
        cat_distribution = Categorical(probs)
        action = cat_distribution.sample()
        return action.item()
    
    def run_episode(self):
        total_reward = 0
        obs, _ = self.env.reset()
        for t in range(self.n_steps):
            action = self.select_action(obs)
            obs, reward, done, *_ = self.env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward
    
    def calc_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long().unsqueeze(1)
        rewards = torch.from_numpy(rewards).float().unsqueeze(1)
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones).float().unsqueeze(1)

        probs = self.policy_net.forward(states).gather(1, actions)
        old_probs = self.old_policy_net.forward(states).gather(1, actions)
        ratio = probs / old_probs

        surrogate_loss = torch.min(ratio * rewards, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * rewards)
        loss = -surrogate_loss.mean()
        return loss
    
    def run_epochs(self):
        print(f"{'-'*25}Training started{'-'*25}")
        avg_rewards = []
        avg_losses = []
        for i_epoch in tqdm(range(self.n_epochs), desc="Epochs"):
            epoch_rewards = []
            epoch_losses = []
            batch = []
            for i_episode in range(self.n_episodes):
                total_reward = self.run_episode()
                epoch_rewards.append(total_reward)
                obs, _ = self.env.reset()
                for t in range(self.n_steps):
                    action = self.select_action(obs)
                    next_obs, reward, done, *_ = self.env.step(action)
                    batch.append((obs, action, reward, next_obs, done))
                    obs = next_obs
                    if done:
                        break
            avg_reward = np.mean(epoch_rewards)
            avg_rewards.append(avg_reward)
            states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
            states = np.stack(states)
            actions = np.stack(actions)
            rewards = np.stack(rewards)
            next_states = np.stack(next_states)
            dones = np.stack(dones)
            batch = [states, actions, rewards, next_states, dones]
            loss = self.calc_loss(batch)
            epoch_losses.append(loss.item())
            avg_loss = np.mean(epoch_losses)
            avg_losses.append(avg_loss)
            if (i_epoch+1)%10 == 0:
                tqdm.write(f'Epoch {i_epoch+1}: Average reward = {avg_reward:.2f} | Average loss = {avg_loss:.4f}')
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        print(f"{'-'*25}Training Finished{'-'*25}")
        return self.policy_net, avg_rewards, avg_losses

    def training_visualisation(self, avg_rewards, avg_losses, fname):
        fig_path = f"training/{fname}.png"
        epochs = range(len(avg_rewards))
        
        # Calculate the rolling average of avg_rewards
        rolling_avg_rewards = rolling_average(avg_rewards, 10)

        # Calculate the rolling average of avg_losses
        rolling_avg_losses = rolling_average(avg_losses, 10)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot avg_rewards in blue and rolling_avg_rewards in red
        axs[0].plot(epochs[9:], avg_rewards[9:], color='blue', label='Avg Rewards')
        axs[0].plot(epochs[9:], rolling_avg_rewards[9:], color='red', label='Rolling Avg Rewards')
        axs[0].axhline(y=200, color='green', linestyle='-', label='Solved')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Avg Rewards')
        axs[0].set_title('Avg Rewards in function of epochs')
        axs[0].legend()

        # Plot avg_losses in blue and rolling_avg_losses in red
        axs[1].plot(epochs[9:], avg_losses[9:], color='blue', label='Avg Losses')
        axs[1].plot(epochs[9:], rolling_avg_losses[9:], color='red', label='Rolling Avg Losses')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Avg Losses')
        axs[1].set_title('Avg Losses in function of epochs')
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(fig_path)
        plt.show()


def rolling_average(data, window_size):
        rolling_sum = [0] * len(data)
        rolling_average = [0] * len(data)
        
        for i in range(window_size):
            rolling_sum[window_size - 1] += data[i]
        
        rolling_average[window_size - 1] = rolling_sum[window_size - 1] / window_size
        
        for i in range(window_size, len(data)):
            rolling_sum[i] = rolling_sum[i - 1] - data[i - window_size] + data[i]
            rolling_average[i] = rolling_sum[i] / window_size
            
        return rolling_average


