import argparse


class Args:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, choices=['MUTAG', 'IMDB-BINARY', 'IMDB-MULTI',
                                                            'PROTEINS', 'PTC_GIN', 'COLLAB', 'ogbg-molhiv',
                                                            "ogbg-moltox21"], default='MUTAG')
        parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
        parser.add_argument('--epochs', type=int, default=350, help='maximum number of epochs')
        parser.add_argument('--configuration', type=str, default="c3",
                            choices=["c1", "c2", "c3", "c4", "sign", "sgcn", 'GIN_ogb', 'GCN_ogb', 'DropGIN_ogb',
                                     'DropGCN_ogb'],
                            help='which configuration to be used')
        args = parser.parse_args()

        self.dataset = args.dataset
        self.seed = args.seed
        self.epochs = args.epochs
        self.configuration = args.configuration
