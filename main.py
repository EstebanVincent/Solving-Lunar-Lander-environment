import argparse

from train import DiscoverEnv, Train
from eval import Demo, Eval


def demo(render):
    """
    No randomisation and no wind
    """

    model = Demo(render)
    model.evaluate()


def discover_env(model_version, n_epochs):
    """
    Train to get the synamic parameters from the first 10 steps in freefall
    """
    discovery = DiscoverEnv(model_version, n_epochs)
    discovery.train()


def train(models_version):
    """
    Train the actor and critic networks to solve a randomized env
    Use the trained discovery network to get the dynamic parameters
    """
    rl = Train(models_version)
    rl.train()


def evaluate(models_version, render):
    """
    Evaluate the actor model
    Use the trained discovery network to get the dynamic parameters
    """
    model = Eval(models_version)
    model.evaluate(render=render)


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=100_000)
    parser.add_argument('--discover_env', default=None)
    parser.add_argument('--train', default=None)
    parser.add_argument('--evaluate', default=None)
    parser.add_argument('--demo', default=None)
    args = parser.parse_args()

    if args.discover_env:
        discover_env(args.analyse, args.n_epochs)

    if args.train:
        train(args.train)

    if args.evaluate:
        evaluate(args.evaluate, render=args.render)

    if args.demo:
        demo(render=args.render)


if __name__ == '__main__':
    main()
