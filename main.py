import argparse

from train import DemoTrain, DiscoverTrain, Train
from eval import DemoEval, DiscoverEval, Eval


def demo_train(m_version, n_epochs):
    """
    No randomisation and no wind
    """
    rl = DemoTrain(m_version, n_epochs)
    rl.train()


def demo_eval(render):
    """
    No randomisation and no wind
    """

    model = DemoEval(render)
    model.evaluate()


def discover_train(d_version, n_epochs):
    """
    Train to get the synamic parameters from the first 10 steps in freefall
    """
    discovery = DiscoverTrain(d_version, n_epochs)
    discovery.train()


def discover_eval(d_version):
    discovery = DiscoverEval(d_version)
    discovery.evaluate()


def train(d_version, m_version, n_epochs):
    """
    Train the actor and critic networks to solve a randomized env
    Use the trained discovery network to get the dynamic parameters
    """
    rl = Train(d_version, m_version, n_epochs)
    rl.train()


def evaluate(d_version, m_version, render):
    """
    Evaluate the actor model
    Use the trained discovery network to get the dynamic parameters
    """
    model = Eval(d_version, m_version)
    model.evaluate(render=render)


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--m_version')
    parser.add_argument('--d_version')

    parser.add_argument('--n_epochs', type=int, default=100_000)

    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discover_train', action="store_true")
    parser.add_argument('--discover_eval', action="store_true")
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--evaluate', action="store_true")
    parser.add_argument('--demo_train', action="store_true")
    parser.add_argument('--demo_eval', action="store_true")
    args = parser.parse_args()

    if args.demo_train:
        demo_train(args.m_version, args.n_epochs)

    if args.demo_eval:
        demo_eval(render=args.render)

    if args.discover_train:
        discover_train(args.d_version, args.n_epochs)

    if args.discover_eval:
        discover_eval(args.d_version)

    if args.train:
        train(args.d_version, args.m_version, args.n_epochs)

    if args.evaluate:
        evaluate(args.d_version, args.m_version, render=args.render)


if __name__ == '__main__':
    main()
