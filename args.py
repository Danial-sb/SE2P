import argparse


class Args:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, choices=['MUTAG', 'IMDB-BINARY', 'IMDB-MULTI',
                                                            'PROTEINS', 'PTC_GIN', 'COLLAB', 'ogbg-molhiv',
                                                            "ogbg-moltox21"], default='MUTAG')
        parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
        parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size of the model')
        parser.add_argument('--L', type=int, default=2, help='(virtual) number of diffusion layers')
        parser.add_argument('--epochs', type=int, default=350, help='maximum number of epochs')
        parser.add_argument('--activation', type=str, default='ELU', choices=['ELU', 'ReLU'])
        parser.add_argument('--batch_norm', type=bool, default=True, help='whether to use batch normalization')
        parser.add_argument('--dropout', type=float, default=0.5, choices=[0.0, 0.5], help='dropout rate')
        parser.add_argument('--configuration', type=str, default="c2",
                            choices=["c1", "c2", "c3", "c4", "sign", "sgcn", 'GIN', 'GCN', 'DropGIN', 'DropGCN'],
                            help='which configuration to use')
        parser.add_argument('--N_mlp', type=int, default=2, help='Number of layers in the last MLP')
        parser.add_argument('--N_pool', type=int, default=4, help='Number of layers in MLP of the POOL function')
        parser.add_argument('--Ds_im', type=int, default=2, help='Number of layers in inner MLP of the DeepSet of the MERGE function')
        parser.add_argument('--Ds_om', type=int, default=2, help='Number of layers in outer MLP of the Deepset of the MERGE function')
        parser.add_argument('--Ds_ic', type=int, default=2, help='Number of layers in inner MLP of the DeepSet of the COMBINE function')
        parser.add_argument('--Ds_oc', type=int, default=2, help='Number of layers in outer MLP of the Deepset of the COMBINE function')
        parser.add_argument('--hidden_dim', type=int, default=32, help='hidden dimensionality of all the layers')
        parser.add_argument('--graph_pooling', type=str, default='sum', choices=['sum', 'attention_agg'])

        args = parser.parse_args()

        self.dataset = args.dataset
        self.seed = args.seed
        self.lr = args.lr
        self.epochs = args.epochs
        self.activation = args.activation
        self.batch_norm = args.batch_norm
        self.dropout = args.dropout
        self.graph_pooling = args.graph_pooling
        self.N_mlp = args.N_mlp
        self.N_pool = args.N_pool
        self.Ds_im = args.Ds_im
        self.Ds_om = args.Ds_om
        self.Ds_ic = args.Ds_ic
        self.Ds_oc = args.Ds_oc
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.configuration = args.configuration
        self.L = args.L
