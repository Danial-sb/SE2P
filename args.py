import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['MUTAG', 'IMDB-BINARY', 'IMDB-MULTI',
                                                        'PROTEINS', 'PTC_GIN', 'COLLAB', 'ogbg-molhiv',
                                                        "ogbg-moltox21"], default='MUTAG')
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    # parser.opt_list('--batch_size', type=int, default=32, tunable=True, choices=[32, 64],
    #                 help='batch size of the model')
    parser.add_argument('--L', type=int, default=3, help='(virtual) number of diffusion layers')
    parser.add_argument('--epochs', type=int, default=350, help='maximum number of epochs')
    parser.add_argument('--activation', type=str, default='ELU', choices=['ELU', 'ReLU'])
    parser.add_argument('--batch_norm', type=bool, default=True, help='whether to use batch normalization')
    # parser.opt_list('--dropout', type=float, default=0.5, tunable=True, choices=[0.0, 0.5], help='dropout rate')
    parser.add_argument('--configuration', type=str, default="c3",
                        choices=["c1", "c2", "c3", "c4", "sign", "sgcn", 'GIN', 'GCN', 'DropGIN', 'DropGCN'],
                        help='which configuration to use')
    # parser.opt_list('--N_mlp', type=int, default=2, tunable=True, choices=[1, 2, 3],
    #                 help='Number of hidden layers in the last MLP')
    # parser.add_argument('--N_pool', type=int, default=2, tunable=True, choices=[1, 2, 3],
    #                 help='Number of hidden layers in MLP of the POOL function')
    # parser.add_argument('--Ds_im', type=int, default=0, tunable=True, choices=[1, 2, 3],
    #                 help='Number of hidden layers in inner MLP of the DeepSet of the MERGE function')
    # parser.add_argument('--Ds_om', type=int, default=0, tunable=True, choices=[1, 2, 3],
    #                 help='Number of hidden layers in outer MLP of the Deepset of the MERGE function')
    # parser.add_argument('--Ds_ic', type=int, default=0, tunable=True, choices=[1, 2, 3],
    #                 help='Number of hidden layers in inner MLP of the DeepSet of the COMBINE function')
    # parser.add_argument('--Ds_oc', type=int, default=0, tunable=True, choices=[1, 2, 3],
    #                 help='Number of hidden layers in outer MLP of the Deepset of the COMBINE function')
    # parser.opt_list('--hidden_dim', type=int, default=128, tunable=True, choices=[16, 32, 64],
    #                 help='hidden dimensionality of all the layers')
    parser.add_argument('--graph_pooling', type=str, default='sum', choices=['sum', 'attention_agg'])

    args = parser.parse_args()
    return args
