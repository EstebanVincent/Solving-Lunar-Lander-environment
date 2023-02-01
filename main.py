import argparse
import gym
import torch
import torch.optim as optim
import numpy as np
from PPO import PolicyNet, PPO
from tqdm import tqdm

def train(fname):
    env = gym.make("LunarLander-v2")
    ppo = PPO(env, env.observation_space.shape[0], 64, env.action_space.n)
    model, avg_rewards, avg_losses = ppo.run_epochs()
    
    torch.save(model.state_dict(), fname)
    env.close()

def evaluate(fname, env=None, n_episodes=10, max_steps_per_episode=500, render=False):
    env = gym.make("LunarLander-v2")
    if render:
        env = gym.make('LunarLander-v2', render_mode='human')
    print(f"{'-'*25}Evaluation started{'-'*25}")
    ppo = PPO(env, env.observation_space.shape[0], 64, env.action_space.n, fname=fname)

    rewards = []
    for episode in tqdm(range(n_episodes), desc='Evaluate'):
        total_reward = 0
        steps = 0
        done = False
        obs, _ = env.reset()
        
        for step in range(max_steps_per_episode):
            steps += 1
            action = ppo.select_action(obs)
            next_obs, reward, done, *_ = env.step(action)
            if render: env.render()
            total_reward += reward
            obs = next_obs
            if done: break
        tqdm.write(f"Ep {episode + 1} : Reward = {total_reward} | Steps = {steps}")
        rewards.append(total_reward)
    print('Mean Reward:', np.mean(rewards))
    print(f"{'-'*25}Evaluation finished{'-'*25}")
    env.close()


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