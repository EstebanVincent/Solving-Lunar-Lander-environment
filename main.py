import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time
from datetime import timedelta


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

def select_action(state, policy_net):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy_net(state)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item()

def run_episode(env, policy_net, episode_len=100):
    total_reward = 0
    obs, _ = env.reset()
    for t in range(episode_len):
        action = select_action(obs, policy_net)
        obs, reward, done, *_ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

def calc_loss(batch, policy_net, old_policy_net, epsilon=0.1):
    states, actions, rewards, next_states, dones = batch
    states = torch.from_numpy(states).float()
    actions = torch.from_numpy(actions).long().unsqueeze(1)
    rewards = torch.from_numpy(rewards).float().unsqueeze(1)
    next_states = torch.from_numpy(next_states).float()
    dones = torch.from_numpy(dones).float().unsqueeze(1)

    probs = policy_net.forward(states).gather(1, actions)
    old_probs = old_policy_net.forward(states).gather(1, actions)
    ratio = probs / old_probs

    surrogate_loss = torch.min(ratio * rewards, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * rewards)
    loss = -surrogate_loss.mean()
    return loss

def try_policy(fname, env=None, n_episodes=10, max_steps_per_episode=200, render=False):
    env = gym.make('LunarLander-v2', render_mode='human')
    policy_net = PolicyNet(env.observation_space.shape[0], 64, env.action_space.n)
    policy_net.load_state_dict(torch.load(fname))

    rewards = []
    for episode in range(n_episodes):
        total_reward = 0
        done = False
        obs, _ = env.reset()
        
        for i in range(max_steps_per_episode):
            action = select_action(obs, policy_net)
            obs, reward, done, *_ = env.step(action)
            if render: env.render()
            total_reward += reward
            s = obs
            if done: break
        
        rewards.append(total_reward)
    print('Mean Reward:', np.mean(rewards))
    
#train
if __name__ != '__main__':
    env = gym.make("LunarLander-v2")
    policy_net = PolicyNet(env.observation_space.shape[0], 64, env.action_space.n)
    old_policy_net = PolicyNet(env.observation_space.shape[0], 64, env.action_space.n)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    avg_rewards = []
    avg_losses = []
    start = time()
    for i_epoch in range(1000):#number of epochs
        epoch_rewards = []
        epoch_losses = []
        batch = []
        for i_episode in range(10): #number of episodes
            total_reward = run_episode(env, policy_net)
            epoch_rewards.append(total_reward)
            obs, _ = env.reset()
            for t in range(100): # number of steps a augmenter pour changer la stagnation
                action = select_action(obs, policy_net)
                next_obs, reward, done, *_ = env.step(action)
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

        loss = calc_loss(batch, policy_net, old_policy_net)
        epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        avg_losses.append(avg_loss)
        if (i_epoch+1)%10 == 0:
            elapsed = str(timedelta(seconds=time() - start)).split(".")[0]
            print(f'Epoch {i_epoch+1}: Average reward = {avg_reward:.2f} | Average loss = {avg_loss:.4f} | Time elapsed = {elapsed}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        old_policy_net.load_state_dict(policy_net.state_dict())
    torch.save(policy_net.state_dict(), 'models/model_PPO_v1.pkl')


#test render
if __name__ == '__main__':
    try_policy('models/model_PPO_v1.pkl', render=True)