import argparse
import gym
import torch
import torch.optim as optim
import numpy as np
from PPO import PolicyNet, run_epochs, select_action

def train(fname):
    env = gym.make("LunarLander-v2")
    policy_net = PolicyNet(env.observation_space.shape[0], 64, env.action_space.n)
    old_policy_net = PolicyNet(env.observation_space.shape[0], 64, env.action_space.n)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    model, avg_rewards, avg_losses = run_epochs(env, policy_net, old_policy_net, optimizer)
    
    torch.save(model.state_dict(), fname)

def evaluate(fname, env=None, n_episodes=10, max_steps_per_episode=200, render=False):
    env = gym.make('LunarLander-v2', render_mode='human')
    policy_net = PolicyNet(env.observation_space.shape[0], 64, env.action_space.n)
    policy_net.load_state_dict(torch.load(fname))

    rewards = []
    for episode in range(n_episodes):
        total_reward = 0
        done = False
        obs, _ = env.reset()
        
        for step in range(max_steps_per_episode):
            action = select_action(obs, policy_net)
            obs, reward, done, *_ = env.step(action)
            if render: env.render()
            total_reward += reward
            s = obs
            if done: break
        
        rewards.append(total_reward)
    print('Mean Reward:', np.mean(rewards))


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', default=None)
    parser.add_argument('-e', '--evaluate', default=None)
    args = parser.parse_args()

    if args.train is not None:
        train(args.train)

    if args.evaluate:
        evaluate(args.evaluate, render=args.render)

if __name__ == '__main__':
    main()