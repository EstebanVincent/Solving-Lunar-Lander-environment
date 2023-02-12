import argparse
import gym

from ppo.ppo import Agent, Analysis


def analyse(model_version, n_epochs):
    analyse = Analysis(model_version, n_epochs)
    analyse.train()


def train(models_version):
    env = gym.make("LunarLander-v2")
    agent = Agent(env, models_version)
    agent.train()
    env.close()


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
    parser.add_argument('-a', '--analyse', default=None)
    parser.add_argument('-t', '--train', default=None)
    parser.add_argument('-e', '--evaluate', default=None)
    args = parser.parse_args()

    if args.analyse:
        analyse(args.analyse, args.n_epochs)

    if args.train:
        train(args.train)

    if args.evaluate:
        evaluate(args.evaluate, render=args.render)


if __name__ == '__main__':
    main()
