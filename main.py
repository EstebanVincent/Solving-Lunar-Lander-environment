import argparse
import gymnasium as gym
from ppo.networks import ActorNetwork
import torch
import numpy as np

from ppo.ppo import Agent, Analysis


def demo_no_rand(render):
    env = gym.make("LunarLander-v2")
    if render:
        env = gym.make('LunarLander-v2', render_mode='human')
    print(f"{'-'*25}Evaluation started{'-'*25}")
    model = ActorNetwork(8, 4, torch.optim.Adam, 1e-3)
    model.load_state_dict(torch.load("model/actor_model.pkl"))
    scores = []
    for episode in range(10):
        state, _ = env.reset()
        score = 0
        while True:
            state = torch.from_numpy(state).float()
            action, _ = model.select_action(state)
            next_state, reward, done, *_ = env.step(action)
            score += reward
            if done:
                break
            state = next_state
        scores.append(score)
    print(f"{'-'*25}Evaluation finished{'-'*25}")
    print('Mean Score:', np.mean(scores))
    env.close()


def analyse(model_version, n_epochs):
    analyse = Analysis(model_version, n_epochs)
    analyse.train()


def train(models_version):
    agent = Agent(models_version)
    agent.train()


def evaluate(models_version, render):
    env = gym.make("LunarLander-v2")
    if render:
        env = gym.make('LunarLander-v2', render_mode='human')
    agent = Agent(env, models_version, existing_model=True)
    agent.evaluate()
    env.close()


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=100_000)
    parser.add_argument('--analyse', default=None)
    parser.add_argument('--train', default=None)
    parser.add_argument('--evaluate', default=None)
    parser.add_argument('--demo', default=None)
    args = parser.parse_args()

    if args.analyse:
        analyse(args.analyse, args.n_epochs)

    if args.train:
        train(args.train)

    if args.evaluate:
        evaluate(args.evaluate, render=args.render)

    if args.demo:
        evaluate(render=args.render)


if __name__ == '__main__':
    main()
