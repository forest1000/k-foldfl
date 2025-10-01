import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=5, help="the number of folds")
    parser.add_argument('--num_clients', type=int, default=5, help='the number of clients')
    parser.add_argument('--rounds', type=int, default=10, help="the number of rounds in federated learning")
    parser.add_argument('--epochs', type=int, default=100, help='local epochs in each round')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--gpu', type=str, default='1', help='whether use gpu')
    parser.add_argument('--beta', type=int, default=100, help='The parameter of the Dirichlet distribution that controls the data distribution skew among clients (a smaller value results in a more non-IID distribution)')
    parser.add_argument('--fraction_fit', type=1.0, default=1.0, help='Client participation rate')
    args, unknown = parser.parse_known_args()
    return args

args = args_parser()

def args2cfg(cfg, args):
    cfg.update(vars(args))
    return cfg