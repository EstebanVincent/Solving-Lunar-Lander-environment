import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# Define the policy network
class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x


class PPO:
    def __init__(self, env, input_size, hidden_size, output_size, fname = None, n_epochs=1000, n_episodes=10, n_steps=500, epsilon=0.1, lr=0.001):
        self.env = env
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.policy_net = PolicyNet(self.input_size, self.hidden_size, self.output_size)

        if fname is None:
            self.n_epochs = n_epochs
            self.n_episodes = n_episodes
            self.n_steps = n_steps
            self.epsilon = epsilon
            self.lr = lr
            
            self.old_policy_net = PolicyNet(self.input_size, self.hidden_size, self.output_size)
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        else :
            self.policy_net.load_state_dict(torch.load(fname))

        
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_net(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
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
        print(f'Final results : Final reward = {avg_rewards[-1]:.2f} | Final loss = {avg_losses[-1]:.4f}')
        return self.policy_net, avg_rewards, avg_losses



    


