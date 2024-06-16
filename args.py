import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['MUTAG', 'IMDB-BINARY', 'IMDB-MULTI',
                                                        'PROTEINS', 'PTC_GIN', 'COLLAB', 'ogbg-molhiv',
                                                        "ogbg-moltox21"], default='MUTAG')
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, choices=[32, 64, 256],
                        help='batch size of the model')
    parser.add_argument('--L', type=int, default=3, choices=[2, 3], help='(virtual) number of diffusion layers')
    parser.add_argument('--epochs', type=int, default=350, help='maximum number of epochs')
    parser.add_argument('--activation', type=str, default='ELU', choices=['ELU', 'ReLU'])
    parser.add_argument('--batch_norm', type=bool, default=True, help='whether to use batch normalization')
    parser.add_argument('--dropout', type=float, default=0.5, choices=[0.0, 0.5], help='dropout rate')
    parser.add_argument('--configuration', type=str, default="c2",
                        choices=["c1", "c2", "c3", "c4", "sign", "sgcn", 'GIN', 'GCN', 'DropGIN', 'DropGCN'],
                        help='which configuration to use')
    parser.add_argument('--n_f', type=int, default=1, choices=[1],
                        help='Number of hidden layers in the last MLP')
    parser.add_argument('--n_p', type=int, default=3, choices=[1, 2, 3],
                        help='Number of hidden layers in MLP of the POOL function')
    parser.add_argument('--ds_mi', type=int, default=1, choices=[0, 1, 2],
                        help='Number of hidden layers in inner MLP of the DeepSet of the MERGE function')
    parser.add_argument('--ds_mo', type=int, default=1, choices=[0, 1, 2],
                        help='Number of hidden layers in outer MLP of the Deepset of the MERGE function')
    parser.add_argument('--ds_ci', type=int, default=1, choices=[0, 1, 3],
                        help='Number of hidden layers in inner MLP of the DeepSet of the COMBINE function')
    parser.add_argument('--ds_co', type=int, default=2, choices=[0, 1, 2, 3],
                        help='Number of hidden layers in outer MLP of the Deepset of the COMBINE function')
    parser.add_argument('--hidden_dim', type=int, default=32, choices=[16, 32, 64],
                        help='hidden dimensionality of all the layers')
    parser.add_argument('--graph_pooling', type=str, default='sum', choices=['sum', 'attention_agg'])

    args = parser.parse_args()
    return args
