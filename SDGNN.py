import torch
import os
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import RandomSampler
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter
from torch_geometric.utils import degree
from torch_geometric.transforms import BaseTransform
from torch.nn.functional import pad
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader.dataloader import Collater
import time
import argparse
import os.path as osp
from torch_geometric.data import Data, InMemoryDataset
import wandb
from ptc_dataset import PTCDataset
from torch_geometric.nn.aggr import AttentionalAggregation
from mlp import MLP
from torch_geometric.nn.inits import reset

sweep_config = {
    "method": "grid",
    "metric": {"name": "test_acc", "goal": "maximize"},
    "parameters": {
        "lr": {"values": [0.01]},
        # "num_layers": {"values": [4]},
        "batch_norm": {"values": [True]},
        "batch_size": {"values": [32, 64]},
        "dropout": {"values": [0.0, 0.5]},
        "normalization": {"values": ["After"]},
        "k": {"values": [2, 3]},  # TODO specify for which was 2 and which was 3. pro(3) ptc(3)
        "sum_or_cat": {"values": ["cat"]},
        "decoder_layers": {"values": [2, 3]},
        "activation": {"values": ["ReLU"]},
        "hidden_dim": {"values": [16, 32, 64]}
        # "graph_pooling": {"values": ["sum"]}
    }
}
sweep_id = wandb.sweep(sweep_config, project="C1-IMDBM")


class FeatureDegree(BaseTransform):
    r"""Adds the node degree as one hot encodings to the node features.

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """

    def __init__(self, max_degree, in_degree=False, cat=True):
        self.in_degree = in_degree
        self.cat = cat
        self.max_degree = max_degree

    def __call__(self, data):
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = degree(idx, data.num_nodes, dtype=torch.long)
        deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.long).unsqueeze(-1)
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_degree})'


def get_dataset(args):
    if 'IMDB' in args.dataset:  # IMDB-BINARY or IMDB-MULTI
        class MyFilter(object):
            def __call__(self, data):
                return data.num_nodes <= 70

        class MyPreTransform(object):
            def __call__(self, data):
                data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
                data.x = F.one_hot(data.x, num_classes=69).to(torch.float)
                return data

        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', f'{args.dataset}')
        dataset = TUDataset(
            path,
            name=args.dataset,
            pre_transform=MyPreTransform(),
            pre_filter=MyFilter())

    elif 'MUTAG' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True

        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'MUTAG')
        dataset = TUDataset(path, name='MUTAG', pre_filter=MyFilter())

    elif 'PROTEINS' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return not (data.num_nodes == 7 and data.num_edges == 12) and data.num_nodes < 450

        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PROTEINS')
        dataset = TUDataset(path, name='PROTEINS', pre_filter=MyFilter())

    elif 'ENZYMES' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return data.num_nodes < 95

        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ENZYMES')
        dataset = TUDataset(path, name='ENZYMES', pre_filter=MyFilter())

    elif 'PTC_GIN' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True

        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PTC_GIN')
        dataset = PTCDataset(path, name='PTC')

    elif 'COLLAB' in args.dataset:

        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'COLLAB')
        dataset = TUDataset(path, name='COLLAB', transform=FeatureDegree(max_degree=491, cat=False))

    elif 'NCI1' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True

        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'NCI1')
        dataset = TUDataset(path, name='NCI1', pre_filter=MyFilter())

    elif 'NCI109' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True

        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'NCI109')
        dataset = TUDataset(path, name='NCI109', pre_filter=MyFilter())
    else:
        raise ValueError("Invalid dataset name")

    return dataset


def separate_data(dataset_len, n_splits, seed):
    # Use same splitting/10-fold as GIN paper
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx_list = []
    for idx in skf.split(np.zeros(dataset_len), np.zeros(dataset_len)):
        idx_list.append(idx)
    return idx_list


def get_adj(edge_index, set_diag=True, symmetric_normalize=True):
    # Convert to dense adjacency matrix
    adj = to_dense_adj(edge_index).squeeze()

    if set_diag:
        identity_matrix = torch.eye(adj.shape[0])
        adj = adj + identity_matrix
    if symmetric_normalize:
        D = torch.diag(adj.sum(dim=1))
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diagonal()))
        adj = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
        adj[torch.isnan(adj)] = 0.0

    return adj


def generate_perturbation(adj, p, num_perturbations, seed):
    torch.manual_seed(seed)

    adj = adj.unsqueeze(0).expand(num_perturbations, -1, -1).clone()
    all_adj = [adj[0].clone()]
    adj_perturbation = adj.clone()  # Make a copy of the original adjacency matrix for this perturbation

    for perturbation in range(1, adj.size(0)):  # Loop over perturbations
        drop = torch.bernoulli(torch.ones([adj.size(1)], device=adj.device) * p).bool()

        for idx, val in enumerate(drop):
            if val.item() == 1:
                adj_perturbation[perturbation, idx, :] = 0
                adj_perturbation[perturbation, :, idx] = 0

        all_adj.append(adj_perturbation[perturbation].clone())

    adj_perturbation = torch.stack(all_adj)

    return adj_perturbation


def compute_symmetric_normalized_perturbed_adj(adj_perturbed):  # This is for doing normalization after perturbation
    normalized_adj = []
    for perturbation in range(adj_perturbed.shape[0]):
        # Compute the degree matrix (D) by summing over rows
        D = torch.diag(adj_perturbed[perturbation].sum(dim=1))

        # Compute the inverse square root of the degree matrix (D_inv_sqrt)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diagonal()))

        # Apply symmetric normalization to the adjacency matrix
        adj_normalized = torch.mm(torch.mm(D_inv_sqrt, adj_perturbed[perturbation]), D_inv_sqrt)
        adj_normalized[torch.isnan(adj_normalized)] = 0.0
        normalized_adj.append(adj_normalized)

    all_normalized_adj = torch.stack(normalized_adj)

    return all_normalized_adj


def diffusion(adj_perturbed, feature_matrix, config, args):
    torch.manual_seed(args.seed)
    enriched_feature_matrices = []
    for perturbation in range(adj_perturbed.size(0)):
        # Get the adjacency matrix for this perturbation
        adj_matrix = adj_perturbed[perturbation]
        feature_matrix_for_perturbation = feature_matrix.clone()

        internal_diffusion = [feature_matrix_for_perturbation.clone()]
        # Perform diffusion for 'k' steps
        for _ in range(config.k):
            # Multiply the adjacency matrix with the perturbed feature matrix for each step
            feature_matrix_for_perturbation = torch.matmul(adj_matrix, feature_matrix_for_perturbation)
            internal_diffusion.append(feature_matrix_for_perturbation.clone())

        if config.sum_or_cat == "sum":
            internal_diffusion = torch.stack(internal_diffusion, dim=0)
            internal_diffusion = torch.sum(internal_diffusion, dim=0)
        elif config.sum_or_cat == "cat":
            internal_diffusion = torch.cat(internal_diffusion, dim=0 if args.agg == "c4" else 1)
        else:
            raise ValueError("AGG in EQ1 should be either cat or sum")

        enriched_feature_matrices.append(internal_diffusion)

    feature_matrices_of_perturbations = torch.stack(enriched_feature_matrices)

    return feature_matrices_of_perturbations


def diffusion_sgcn(adj_perturbed, feature_matrix, config, seed):
    torch.manual_seed(seed)
    enriched_feature_matrices = []
    for perturbation in range(adj_perturbed.size(0)):
        # Get the adjacency matrix for this perturbation
        adj_matrix = adj_perturbed[perturbation]
        feature_matrix_for_perturbation = feature_matrix.clone()

        # Perform diffusion for 'k' steps
        for _ in range(config.k):
            feature_matrix_for_perturbation = torch.matmul(adj_matrix, feature_matrix_for_perturbation)

        enriched_feature_matrices.append(feature_matrix_for_perturbation)

    feature_matrices_of_perturbations = torch.stack(enriched_feature_matrices)

    return feature_matrices_of_perturbations


class EnrichedGraphDataset(InMemoryDataset):
    def __init__(self, root, name, dataset, p, num_perturbations, config, args):
        super(EnrichedGraphDataset, self).__init__(root, transform=None, pre_transform=None)
        self.name = name
        self.p = p
        self.num_perturbations = num_perturbations

        if self._processed_file_exists():
            print("Dataset was already in memory.")
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print("Preprocessing ...")
            self.data_list = self.process_dataset(dataset, config, args)
            self.data, self.slices = torch.load(self.processed_paths[0])

    # def pad_data(self, feature_matrix, adj):
    #     num_nodes = feature_matrix.size(0)
    #
    #     # Pad feature matrix if necessary
    #     if num_nodes < self.max_nodes:
    #         pad_size = self.max_nodes - num_nodes
    #         feature_matrix = torch.cat([feature_matrix, torch.zeros(pad_size, feature_matrix.size(1))], dim=0)
    #
    #     # Pad adjacency if necessary
    #     if adj.size(0) < self.max_nodes:
    #         pad_size = self.max_nodes - adj.size(0)
    #         zeros_pad = torch.zeros(pad_size, adj.size(1))
    #         adj = torch.cat([adj, zeros_pad], dim=0)
    #         adj = torch.cat([adj, torch.zeros(adj.size(0), pad_size)], dim=1)
    #
    #     return feature_matrix, adj

    def process_dataset(self, dataset, config, args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        enriched_dataset = []
        # final_feature_of_graph = None
        counter = 0
        for data in dataset:
            edge_index = data.edge_index
            feature_matrix = data.x.clone().float()  # Converted to float for ogb
            num_nodes = feature_matrix.size(0)

            if args.agg not in ["c1", "c4", "deepset", "mean", "sign", "sgcn"]:
                raise ValueError("Invalid aggregation method specified")

            if config.normalization not in ["Before", "After"]:
                raise ValueError("Invalid normalization configuration")

            if args.agg == "deepset" and config.normalization == "Before":
                adj = get_adj(edge_index, set_diag=False, symmetric_normalize=True)
                feature_matrix, adj = self.pad_data(feature_matrix, adj)
                # if adj.size(0) != feature_matrix.size(0): # for ogb
                #     counter = counter + 1
                #     max_size = max(adj.size(0), feature_matrix.size(0))
                #     pad_amount = max_size - adj.size(0)
                #     adj = pad(adj, (0, pad_amount, 0, pad_amount), mode='constant', value=0)
                perturbed_adj = generate_perturbation(adj, self.p, self.num_perturbations, args.seed)
                feature_matrices_of_perts = diffusion(perturbed_adj, feature_matrix, config, args)
                final_feature_of_graph = feature_matrices_of_perts.view(-1, feature_matrices_of_perts.size(
                    -1))  # remove view if wanted to use previous

            elif args.agg == "deepset" or args.agg == "c4" and config.normalization == "After":
                adj = get_adj(edge_index, set_diag=False,
                              symmetric_normalize=False)  # set diag here can be false or true
                if adj.size(0) != feature_matrix.size(0):  # This is for ogb
                    counter = counter + 1
                    max_size = max(adj.size(0), feature_matrix.size(0))
                    pad_amount = max_size - adj.size(0)
                    adj = pad(adj, (0, pad_amount, 0, pad_amount), mode='constant', value=0)
                # if num_nodes < self.max_nodes:
                #     pad_size = self.max_nodes - num_nodes
                #     feature_matrix = torch.cat([feature_matrix, torch.zeros(pad_size, feature_matrix.size(1))], dim=0)
                #
                # if adj.size(0) < self.max_nodes:
                #     pad_size = self.max_nodes - adj.size(0)
                #     zeros_pad = torch.zeros(pad_size, adj.size(1))
                #     adj = torch.cat([adj, zeros_pad], dim=0)
                #     adj = torch.cat([adj, torch.zeros(adj.size(0), pad_size)], dim=1)

                # feature_matrix, adj = self.pad_data(feature_matrix, adj)
                perturbed_adj = generate_perturbation(adj, self.p, self.num_perturbations, args.seed)
                normalized_adj = compute_symmetric_normalized_perturbed_adj(perturbed_adj)
                if torch.isnan(normalized_adj).any():
                    raise ValueError("NaN values encountered in normalized adjacency matrices.")
                feature_matrices_of_perts = diffusion(normalized_adj, feature_matrix, config, args)
                final_feature_of_graph = feature_matrices_of_perts.view(-1, feature_matrices_of_perts.size(-1))

            elif args.agg == "mean" and config.normalization == "Before":
                adj = get_adj(edge_index, set_diag=False, symmetric_normalize=True)
                if adj.size(0) != feature_matrix.size(0):
                    counter = counter + 1
                    max_size = max(adj.size(0), feature_matrix.size(0))
                    pad_amount = max_size - adj.size(0)
                    adj = pad(adj, (0, pad_amount, 0, pad_amount), mode='constant', value=0)
                perturbed_adj = generate_perturbation(adj, self.p, self.num_perturbations, args.seed)
                feature_matrices_of_perts = diffusion(perturbed_adj, feature_matrix, config, args)
                final_feature_of_graph = feature_matrices_of_perts.mean(dim=0).clone()

            elif args.agg == "mean" or args.agg == "c1" and config.normalization == "After":
                adj = get_adj(edge_index, set_diag=False, symmetric_normalize=False)
                if adj.size(0) != feature_matrix.size(0):
                    counter = counter + 1
                    max_size = max(adj.size(0), feature_matrix.size(0))
                    pad_amount = max_size - adj.size(0)
                    adj = pad(adj, (0, pad_amount, 0, pad_amount), mode='constant', value=0)
                perturbed_adj = generate_perturbation(adj, self.p, self.num_perturbations, args.seed)
                normalized_adj = compute_symmetric_normalized_perturbed_adj(perturbed_adj)
                if torch.isnan(normalized_adj).any():
                    raise ValueError("NaN values encountered in normalized adjacency matrices.")
                feature_matrices_of_perts = diffusion(normalized_adj, feature_matrix, config, args)
                final_feature_of_graph = feature_matrices_of_perts.mean(dim=0).clone()

            elif args.agg == "sign":
                adj = get_adj(edge_index, set_diag=False, symmetric_normalize=True)
                adj = adj.unsqueeze(0).expand(1, -1, -1).clone()
                final_feature_of_graph = diffusion(adj, feature_matrix, config, args)
                final_feature_of_graph = final_feature_of_graph.squeeze()

            elif args.agg == "sgcn":
                adj = get_adj(edge_index, set_diag=False, symmetric_normalize=True)
                adj = adj.unsqueeze(0).expand(1, -1, -1).clone()
                final_feature_of_graph = diffusion_sgcn(adj, feature_matrix, config, args.seed)
                final_feature_of_graph = final_feature_of_graph.squeeze()

            else:
                raise ValueError("Error in choosing hyper parameters")

            enriched_data = Data(x=final_feature_of_graph, edge_index=edge_index, y=data.y)
            enriched_dataset.append(enriched_data)

        print(counter)
        data, slices = self.collate(enriched_dataset)
        path = self.processed_paths[0]
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        torch.save((data, slices), self.processed_paths[0])

    def _processed_file_exists(self):
        return os.path.exists(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = 'processed'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_file_names(self):
        return ['data.pt']


# class DeepSet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_perturbations):
#         super(DeepSet, self).__init__()
#
#         self.num_perturbations = num_perturbations
#         # MLP for individual perturbations
#         self.mlp_perturbation = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ELU()
#         )
#
#         # MLP for aggregation
#         self.mlp_aggregation = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.ELU()
#         )
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 nn.init.constant_(module.bias, 0)
#
#     def forward(self, input_data):
#         # Apply MLP to each perturbation
#         # print(input_data.shape)
#         perturbation_outputs = torch.stack(
#             [self.mlp_perturbation(input_data[:, i, :]) for i in range(self.num_perturbations)])
#         # print(perturbation_outputs.shape)
#         # Sum over perturbations
#         aggregated_output = torch.sum(perturbation_outputs, dim=0)
#         # print(aggregated_output.shape)
#         # Apply MLP to the aggregated output
#         final_output = self.mlp_aggregation(aggregated_output)
#         return final_output
#
#
# class SDGNN_DeepSet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout, num_perturbations):
#         super(SDGNN_DeepSet, self).__init__()
#
#         self.deepset_aggregator = DeepSet(input_dim, hidden_dim, num_perturbations)
#
#         self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
#         self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
#         self.activation = nn.ELU()
#         self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.bn2 = nn.BatchNorm1d(hidden_dim)
#         self.linear3 = nn.Linear(hidden_dim, output_dim)
#         self.dropout = dropout
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.linear1.reset_parameters()
#         self.linear2.reset_parameters()
#         self.linear3.reset_parameters()
#
#     def forward(self, data):
#         x = data.x  # [num_perturbation, num_nodes, num_features]
#         batch = data.batch
#         # print(f'input: {x.shape}')
#         aggregated_features = self.deepset_aggregator(x)
#
#         x = self.bn1(self.linear1(aggregated_features))
#         x = self.activation(x)
#
#         x = self.bn2(self.linear2(x))
#         x = self.activation(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)
#
#         x = global_add_pool(x, batch)
#
#         x = self.linear3(x)
#
#         return F.log_softmax(x, dim=-1)

# this commented now!
# class DeepSet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_perturbations, device):
#         super(DeepSet, self).__init__()
#
#         self.hidden_size = hidden_size
#         # MLP for individual perturbations
#         self.mlp_local = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             nn.ELU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ELU()
#         )
#
#         # MLP for aggregation
#         self.mlp_global = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             nn.ELU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ELU()
#         )
#
#         self.num_perturbations = num_perturbations
#         self.device = device
#
#     def forward(self, data):
#         # batch_size = input_data.size(0)
#
#         x = self.mlp_local(data.x)
#         ptr = data.ptr
#         nodes = (torch.diff(ptr) / self.num_perturbations).to(torch.int16).to(self.device)
#         idx_list = []
#         start = 0
#         for node in nodes:
#             idx = torch.arange(start, start + node).repeat(self.num_perturbations)
#             idx_list.append(idx)
#             start += node
#         idx_cat = torch.cat(idx_list, dim=0).to(self.device)
#         aggregated_output = scatter(x, idx_cat, dim=-2, reduce='sum')
#
#         # x_transformed = x.view(self.num_perturbations, input_data.size(0) // self.num_perturbations,
#         # self.hidden_size)  for batch=1
#
#         # x_transformed = x.view(batch_size // self.num_perturbations, self.num_perturbations, self.hidden_size)
#         # x_transformed = x.view(-1, self.num_perturbations, self.max_nodes, self.hidden_size)
#         # aggregated_output = torch.sum(x_transformed, dim=1)
#         # aggregated_output = aggregated_output.view(-1, self.hidden_size)
#         # aggregated_output = torch.sum(x_transformed, dim=0) va in bara batch=1 bod
#
#         final_output = self.mlp_global(aggregated_output)
#
#         return nodes, final_output
#
#
# class SDGNN_Deepset(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout, num_perturbations, device):
#         super(SDGNN_Deepset, self).__init__()
#
#         self.num_perturbations = num_perturbations
#         self.device = device
#
#         self.deepset_aggregator = DeepSet(input_dim, hidden_dim, num_perturbations, device)
#
#         self.linear1 = nn.Linear(hidden_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.elu = nn.ELU()
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim // 2)
#         self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
#         self.linear3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
#         self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
#         self.linear4 = nn.Linear(hidden_dim // 4, output_dim)
#         self.dropout = dropout
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.reset_parameters()
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.reset_parameters()
#
#     def forward(self, data):
#         # x = data.x
#         # batch = data.batch
#
#         nodes, aggregated_features = self.deepset_aggregator(data)
#         batch_indexing = torch.zeros(aggregated_features.size(0), dtype=torch.long, device=self.device)
#         start_idx = 0
#
#         for idx, boundary in enumerate(nodes):
#             batch_indexing[start_idx:start_idx + boundary] = idx
#             start_idx += boundary
#         # batch = batch.view(-1, self.num_perturbations).to(torch.float).mean(dim=1).long()
#
#         x = self.elu(self.bn1(self.linear1(aggregated_features)))
#
#         x = self.elu(self.bn2(self.linear2(x)))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#
#         x = global_add_pool(x, batch_indexing)
#
#         x = self.elu(self.bn3(self.linear3(x)))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#
#         x = self.linear4(x)
#
#         return F.log_softmax(x, dim=-1)


class Decoder(nn.Module):  # This is for the decoder of the SDGNN C2, C3 and C4.
    def __init__(self, input_size, output_size, num_layers, dropout_rate, config, hidden_factor=2, batch_norm=False,
                 dropout=True):
        super(Decoder, self).__init__()

        hidden_sizes = [input_size // (hidden_factor ** i) for i in range(num_layers)]
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            if config.activation == 'ELU':
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class SDGNN_C1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, decoder_layers, config):
        super(SDGNN_C1, self).__init__()

        # self.decoder = Decoder(input_dim, output_dim, decoder_layers, dropout, config, hidden_factor=1,
        #                        batch_norm=config.batch_norm)

        self.decoder = MLP(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=output_dim,
                           num_layers=decoder_layers, batch_norm="batch_norm" if config.batch_norm else None,
                           dropout=[dropout] * decoder_layers, activation=config.activation)

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.decoder)

    def forward(self, data):
        x = data.x
        batch = data.batch

        x = global_add_pool(x, batch)

        x = self.decoder(x)

        return F.log_softmax(x, dim=-1)


class SDGNN_C2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, num_layers, decoder_layers, config, batch_norm=True):
        super(SDGNN_C2, self).__init__()

        self.num_layers = num_layers
        self.batch_norm = batch_norm

        self.linears = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])

        if self.batch_norm:
            self.bns = nn.ModuleList([
                nn.BatchNorm1d(hidden_dim)
                for _ in range(num_layers)
            ])

        if config.activation == 'ELU':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        self.dropout = dropout

        if self.config.graph_pooling == 'sum':
            self.pool = global_add_pool
        elif self.config.graph_pooling == 'attention_agg':
            self.pool = AttentionalAggregation(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(hidden_dim, 2 * hidden_dim),
                                            torch.nn.BatchNorm1d(2 * hidden_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(2 * hidden_dim, 1)))

        self.decoder = Decoder(hidden_dim, output_dim, decoder_layers, dropout, config)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data):
        x = data.x
        batch = data.batch

        for i in range(self.num_layers):
            x = self.linears[i](x)
            if self.batch_norm:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pool(x, batch)

        x = self.decoder(x)

        return F.log_softmax(x, dim=-1)


class SDGNN_C3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_perturbations, dropout, mlp_local_layers,
                 mlp_global_layers, decoder_layers, device, config):
        super(SDGNN_C3, self).__init__()

        self.hidden_size = hidden_size
        self.config = config
        # MLP for individual perturbations
        self.mlp_local = self.create_mlp(input_size, hidden_size, mlp_local_layers)

        # MLP for aggregation
        self.mlp_global = self.create_mlp(hidden_size, hidden_size, mlp_global_layers)

        if self.config.graph_pooling == 'sum':
            self.pool = global_add_pool
        elif self.config.graph_pooling == 'attention_agg':
            self.pool = AttentionalAggregation(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(hidden_size, 2 * hidden_size),
                                            torch.nn.BatchNorm1d(2 * hidden_size), torch.nn.ReLU(),
                                            torch.nn.Linear(2 * hidden_size, 1)))

        self.decoder = Decoder(hidden_size, output_size, decoder_layers, dropout, config)

        self.num_perturbations = num_perturbations
        self.device = device

        self.reset_parameters()

    def create_mlp(self, input_size, hidden_size, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            if self.config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if self.config.activation == 'ELU':
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            input_size = hidden_size
        return nn.Sequential(*layers)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data):

        x = self.mlp_local(data.x)

        ptr = data.ptr
        nodes = (torch.diff(ptr) / self.num_perturbations).to(torch.long).to(self.device)
        idx_list = []
        start = 0
        for node in nodes:
            idx = torch.arange(start, start + node).repeat(self.num_perturbations)
            idx_list.append(idx)
            start += node
        idx_cat = torch.cat(idx_list, dim=0).to(self.device)
        aggregated_output = scatter(x, idx_cat, dim=-2, reduce='sum')

        ds_output = self.mlp_global(aggregated_output)

        batch_indexing = torch.zeros(ds_output.size(0), dtype=torch.long, device=self.device)
        start_idx = 0

        for idx, boundary in enumerate(nodes):
            batch_indexing[start_idx:start_idx + boundary] = idx
            start_idx += boundary

        x = self.pool(ds_output, batch_indexing)

        x = self.decoder(x)

        return F.log_softmax(x, dim=-1)


class SDGNN_C4(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_perturbations, mlp_local_layers,
                 mlp_global_layers, device, config):
        super(SDGNN_C4, self).__init__()

        self.hidden_size = hidden_size
        self.k = config.k
        self.num_perturbations = num_perturbations
        self.device = device

        self.mlp_local_combine = MLP(in_channels=input_size, hidden_channels=hidden_size, out_channels=hidden_size,
                                     num_layers=mlp_local_layers, plain_last=False)

        self.mlp_global_combine = MLP(in_channels=hidden_size, hidden_channels=hidden_size, out_channels=hidden_size,
                                      num_layers=mlp_global_layers, plain_last=False)

        self.mlp_local_merge = MLP(in_channels=hidden_size, hidden_channels=hidden_size, out_channels=hidden_size,
                                   num_layers=mlp_local_layers, plain_last=False)

        self.mlp_global_merge = MLP(in_channels=hidden_size, hidden_channels=hidden_size, out_channels=hidden_size,
                                    num_layers=mlp_global_layers, plain_last=False)

        if config.graph_pooling == 'sum':
            self.pool = global_add_pool
        elif config.graph_pooling == 'attention_agg':
            self.pool = AttentionalAggregation(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(hidden_size, 2 * hidden_size),
                                            torch.nn.BatchNorm1d(2 * hidden_size), torch.nn.ReLU(),
                                            torch.nn.Linear(2 * hidden_size, 1)))

        self.decoder = Decoder(hidden_size, output_size, config.decoder_layers, config.dropout, config)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp_local_combine)
        reset(self.mlp_global_combine)
        reset(self.mlp_local_merge)
        reset(self.mlp_global_merge)
        reset(self.pool)
        reset(self.decoder)

    def forward(self, data):

        ptr = data.ptr
        nodes = (torch.diff(ptr) / (self.num_perturbations * (self.k + 1))).to(torch.long).to(self.device)

        start_comb = 0
        idx_list_all = []
        for node in nodes:
            idx_list_comb = []
            for i in range(self.num_perturbations):
                idx_comb = torch.arange(start_comb, start_comb + node).repeat(self.k + 1)
                idx_list_comb.append(idx_comb)
                start_comb += node

            idx_comb = torch.cat(idx_list_comb, dim=0)
            idx_list_all.append(idx_comb)

        idx_comb = torch.cat(idx_list_all, dim=0).to(self.device)

        x = self.mlp_local_combine(data.x)

        combine_output = scatter(x, idx_comb, dim=-2, reduce='sum')

        x = self.mlp_global_combine(combine_output)

        idx_list_merge = []
        start_merge = 0
        for node in nodes:
            idx_merge = torch.arange(start_merge, start_merge + node).repeat(self.num_perturbations)
            idx_list_merge.append(idx_merge)
            start_merge += node
        idx_merge = torch.cat(idx_list_merge, dim=0).to(self.device)

        x = self.mlp_local_merge(x)

        aggregated_output = scatter(x, idx_merge, dim=-2, reduce='sum')

        batch_indexing = torch.zeros(aggregated_output.size(0), dtype=torch.long, device=self.device)
        start_idx = 0

        for idx, boundary in enumerate(nodes):
            batch_indexing[start_idx:start_idx + boundary] = idx
            start_idx += boundary

        ds_output = self.mlp_global_merge(aggregated_output)

        x = self.pool(ds_output, batch_indexing)

        x = self.decoder(x)

        return F.log_softmax(x, dim=-1)


def train(model, loader, optimizer, device):
    model.train()
    loss_all = 0
    n = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logs = model(data)
        loss = F.nll_loss(logs, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        n += len(data.y)
        optimizer.step()
    return loss_all / n


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset) * 100


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(config=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['MUTAG', 'IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'ENZYMES',
                                                        'PTC_GIN', 'NCI1', 'NCI109', 'COLLAB'], default='IMDB-MULTI',
                        help="Options are ['MUTAG', 'IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'ENZYMES', 'PTC_GIN', "
                             "'NCI1', 'NCI109']")
    # parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--ds_local_layers', type=int, default=2, help='number of local layers in deepset')
    parser.add_argument('--ds_global_layers', type=int, default=2, help='number of global layers in deepset')
    # parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    # parser.add_argument('--dropout', type=float, choices=[0.5, 0.2], default=0.2, help='dropout probability')
    parser.add_argument('--epochs', type=int, default=350, help='maximum number of epochs')
    # parser.add_argument('--min_delta', type=float, default=0.001, help='min_delta in early stopping')
    # parser.add_argument('--patience', type=int, default=100, help='patience in early stopping')
    parser.add_argument('--agg', type=str, default="c1",
                        choices=["c1", "c4", "mean", "deepset", "sign", "sgcn"],
                        help='Method for aggregating the perturbation')
    # parser.add_argument('--normalization', type=str, default='After', choices=['After', 'Before'],
    #                    help='Doing normalization before generation of perturbations or after')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    dataset = get_dataset(args)
    print(dataset)
    n = []
    degs = []
    for g in dataset:
        num_nodes = g.num_nodes
        deg = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
        n.append(num_nodes)
        degs.append(deg.max())
    print(f'Mean Degree: {torch.stack(degs).float().mean()}')
    print(f'Max Degree: {torch.stack(degs).max()}')
    print(f'Min Degree: {torch.stack(degs).min()}')

    mean_n = torch.tensor(n).float().mean().round().long().item()
    max_nodes = torch.tensor(n).float().max().round().long().item()
    min_nodes = torch.tensor(n).float().min().round().long().item()
    print(f'Mean number of nodes: {mean_n}')
    print(f'Max number of nodes: {max_nodes}')
    print(f'Min number of nodes: {min_nodes}')
    print(f'Number of graphs: {len(dataset)}')
    gamma = mean_n
    p = 2 * 1 / (1 + gamma)
    # num_perturbations = round(gamma * np.log10(gamma))
    num_perturbations = gamma
    print(f'Number of perturbations: {num_perturbations}')
    print(f'Sampling probability: {p}')
    current_path = os.getcwd()

    wandb.login()
    with wandb.init(config=config):
        config = wandb.config
        print(args)
        name = f"enriched_{args.dataset}_{args.agg}_{config.k}"
        start_time = time.time()
        enriched_dataset = EnrichedGraphDataset(os.path.join(current_path, 'enriched_dataset'), name, dataset, p=p,
                                                num_perturbations=num_perturbations, config=config, args=args)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done! Time taken: {elapsed_time:.2f} seconds")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        print(f'Device: {device}')
        # seeds_to_test = [args.seed]
        n_splits = 10
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # final_acc = []
        # final_std = []

        # for seed in seeds_to_test:
        # print(f'Seed: {seed}')
        # print("==============")
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # generator = torch.Generator()
        # generator.manual_seed(seed)
        all_validation_accuracies = []
        time_seed = []

        skf_splits = separate_data(len(enriched_dataset), n_splits, args.seed)

        if args.agg == "deepset":
            model = SDGNN_C3(enriched_dataset.num_features, config.hidden_dim,
                             enriched_dataset.num_classes, num_perturbations, config.dropout, args.ds_local_layers,
                             args.ds_global_layers, config.decoder_layers, device, config).to(
                device)  # TODO simplify the args
        elif args.agg == "c1":
            model = SDGNN_C1(enriched_dataset.num_features, config.hidden_dim, enriched_dataset.num_classes,
                             config.dropout, config.decoder_layers, config).to(device)
        elif args.agg == "c4":
            model = SDGNN_C4(enriched_dataset.num_features, config.hidden_dim, enriched_dataset.num_classes,
                             num_perturbations, args.ds_local_layers, args.ds_global_layers, device,
                             config).to(device)
        else:
            model = SDGNN_C2(enriched_dataset.num_features, config.hidden_dim, enriched_dataset.num_classes,
                             config.dropout, config.num_layers, config.decoder_layers, config, config.batch_norm).to(
                device)

        # Iterate through each fold
        for fold, (train_indices, test_indices) in enumerate(skf_splits):
            model.reset_parameters()
            print(f'Fold {fold + 1}/{n_splits}:')
            start_time_fold = time.time()
            # Create data loaders for the current fold
            train_loader = DataLoader(
                enriched_dataset[train_indices.tolist()],
                sampler=RandomSampler(dataset[train_indices.tolist()], replacement=True,
                                      num_samples=int(
                                          len(train_indices.tolist()) * 50 / (
                                                  len(train_indices.tolist()) / config.batch_size)),
                                      generator=generator),
                batch_size=config.batch_size, drop_last=False,
                collate_fn=Collater(follow_batch=[], exclude_keys=[]))

            test_loader = DataLoader(enriched_dataset[test_indices.tolist()], batch_size=config.batch_size,
                                     shuffle=False)

            if fold == 0:
                print(f'Model learnable parameters: {count_parameters(model)}')
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

            time_per_epoch = []
            max_memory_allocated = 0
            max_memory_reserved = 0
            validation_accuracies = []
            # Training loop for the current fold
            for epoch in range(1, args.epochs + 1):
                start_time_epoch = time.time()
                lr = scheduler.optimizer.param_groups[0]['lr']
                train_loss = train(model, train_loader, optimizer, device)
                scheduler.step()

                memory_allocated = torch.cuda.max_memory_allocated(device) // (1024 ** 2)
                memory_reserved = torch.cuda.max_memory_reserved(device) // (1024 ** 2)
                max_memory_allocated = max(max_memory_allocated, memory_allocated)
                max_memory_reserved = max(max_memory_reserved, memory_reserved)

                test_acc = test(model, test_loader, device)
                end_time_epoch = time.time()
                elapsed_time_epoch = end_time_epoch - start_time_epoch
                time_per_epoch.append(elapsed_time_epoch)
                if epoch % 25 == 0 or epoch == 1:
                    print(f'Epoch: {epoch:02d} | TrainLoss: {train_loss:.3f} | Test_acc: {test_acc:.3f} | Time'
                          f'/epoch: {elapsed_time_epoch:.2f} | Memory Allocated: {memory_allocated} MB | Memory '
                          f'Reserved: {memory_reserved} MB | LR: {lr:.6f}')
                wandb.log({"test acc": test_acc})
                validation_accuracies.append(test_acc)

            print(f'Average time per epoch in fold {fold + 1} and seed {args.seed}: {np.mean(time_per_epoch)}')
            print(f'Std time per epoch in fold {fold + 1} and seed {args.seed}: {np.std(time_per_epoch)}')
            all_validation_accuracies.append(torch.tensor(validation_accuracies))
            # Print fold training time
            end_time_fold = time.time()
            elapsed_time_fold = end_time_fold - start_time_fold
            print(f'Time taken for training in seed {args.seed}, fold {fold + 1}: {elapsed_time_fold:.2f} seconds, '
                  f'Max Memory Allocated: {max_memory_allocated} MB | Max Memory Reserved: {max_memory_reserved} MB')
            time_seed.append(elapsed_time_fold)
        print("=" * 30)
        # average_validation_curve = np.mean(all_validation_accuracies, axis=0)
        average_validation_curve = torch.stack(all_validation_accuracies, dim=0)
        acc_mean = average_validation_curve.mean(dim=0)
        best_epoch = acc_mean.argmax().item()
        best_epoch_mean = average_validation_curve[:, best_epoch].mean()
        # max_avg_validation_acc_epoch = np.argmax(average_validation_curve)
        # best_epoch_mean = average_validation_curve[max_avg_validation_acc_epoch]
        # std_at_max_avg_validation_acc_epoch = np.std(
        #     [validation_accuracies[max_avg_validation_acc_epoch] for validation_accuracies in
        #      all_validation_accuracies], ddof=1)
        std_at_max_avg_validation_acc_epoch = average_validation_curve[:, best_epoch].std()

        # final_acc.append(best_epoch_mean)
        # final_std.append(std_at_max_avg_validation_acc_epoch)

        print(f'Epoch {best_epoch + 1} got maximum averaged validation accuracy in seed {args.seed}: '
              f'{best_epoch_mean}')
        print(f'Standard Deviation for the results of epoch {best_epoch + 1} over all the folds '
              f'in seed {args.seed}: {std_at_max_avg_validation_acc_epoch}')
        print(f'Average time taken for each fold in seed {args.seed}: {np.mean(time_seed)}')
        print(f'STD time taken for each fold in seed {args.seed}: {np.std(time_seed)}')

        # print("=" * 30)
        # final_accuracy = np.mean(final_acc)
        # print(f'Test accuracy for all the seeds: {final_accuracy}')
        # print(f'Std for all the seeds: {np.mean(final_std)}')
        wandb.log(
            {"Test Accuracy": best_epoch_mean,
             "Std": std_at_max_avg_validation_acc_epoch
             })


if __name__ == "__main__":
    wandb.agent(sweep_id, main)
