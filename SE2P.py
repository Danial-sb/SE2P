import os
import random
import time
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import RandomSampler
from torch.nn.functional import pad

from sklearn.model_selection import StratifiedKFold

from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import degree, to_dense_adj
from torch_geometric.nn.inits import reset

from torch_scatter import scatter

from args import Args
from datasets import get_dataset



def separate_data(dataset_len: int, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Separate dataset indices into stratified folds.

    Parameters:
    dataset_len (int): Length of the dataset.
    n_splits (int): Number of splits/folds.
    seed (int): Random seed for reproducibility.

    Returns:
    List[Tuple[np.ndarray, np.ndarray]]: List of tuples containing train and test indices for each fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx_list = []

    for idx in skf.split(np.zeros(dataset_len), np.zeros(dataset_len)):
        idx_list.append(idx)
    return idx_list


def get_adj(edge_index: Tensor, set_diag: bool = True, symmetric_normalize: bool = True) -> Tensor:
    """
    Generate a dense adjacency matrix from edge indices with optional diagonal setting and symmetric normalization.

    Parameters:
    edge_index (torch.Tensor): Edge indices of the graph.
    set_diag (bool): If True, set the diagonal to 1. Defaults to True.
    symmetric_normalize (bool): If True, apply symmetric normalization. Defaults to True.

    Returns:
    torch.Tensor: Dense adjacency matrix.
    """
    adj = to_dense_adj(edge_index).squeeze()

    if set_diag:
        adj = adj + torch.eye(adj.size(0), device=adj.device)
    if symmetric_normalize:
        D = torch.diag(adj.sum(dim=1))
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diagonal()))
        adj = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
        adj[torch.isnan(adj)] = 0.0

    return adj


def generate_perturbation(adj: Tensor, p: float, num_perturbations: int, seed: int) -> Tensor:
    """
    Generate perturbed adjacency matrices.

    Parameters:
    adj (torch.Tensor): The original adjacency matrix.
    p (float): Probability of dropping a node.
    num_perturbations (int): Number of perturbations to generate.
    seed (int): Random seed for reproducibility.

    Returns:
    torch.Tensor: Tensor of perturbed adjacency matrices.
    """
    torch.manual_seed(seed)

    adj = adj.unsqueeze(0).expand(num_perturbations, -1, -1).clone()
    all_adj = [adj[0].clone()]

    for perturbation in range(1, num_perturbations):
        drop_mask = torch.bernoulli(torch.full((adj.size(1),), p, device=adj.device)).bool()
        adj_perturbation = adj[perturbation].clone()

        adj_perturbation[drop_mask, :] = 0
        adj_perturbation[:, drop_mask] = 0

        all_adj.append(adj_perturbation)

    all_perturbation = torch.stack(all_adj)

    return all_perturbation


def compute_symmetric_normalized_perturbed_adj(adj_perturbed: Tensor) -> Tensor:
    """
    This function computes the symmetric normalized adjacency matrix for each perturbed adjacency matrix.

    Parameters:
    adj_perturbed (torch.Tensor): A tensor of perturbed adjacency matrices.
                                  The shape of the tensor is (num_perturbations, num_nodes, num_nodes).

    Returns:
    torch.Tensor: A tensor of symmetric normalized adjacency matrices.
                  The shape of the tensor is the same as the input tensor.
    """
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


def diffusion(adj_perturbed: Tensor, feature_matrix: Tensor, args: Any) -> Tensor:
    """
    Perform feature diffusion on a perturbed adjacency matrix.

    Parameters:
    adj_perturbed (torch.Tensor): Tensor of perturbed adjacency matrices (num_perturbations, n, n).
    feature_matrix (torch.Tensor): Tensor of feature matrix (n, d).
    args (object): Arguments object with 'seed' , 'k', and 'configuration' attributes.

    Returns:
    torch.Tensor: Tensor of enriched feature matrices after diffusion.
    """
    torch.manual_seed(args.seed)
    enriched_feature_matrices = []

    cat_dim = 0 if args.configuration == "c4" else 1

    for perturbation in range(adj_perturbed.size(0)):
        # Get the adjacency matrix for this perturbation
        adj_matrix = adj_perturbed[perturbation]
        feature_matrix_for_perturbation = feature_matrix.clone()

        internal_diffusion = [feature_matrix_for_perturbation.clone()]
        # Perform diffusion for 'L' steps
        for _ in range(args.L):
            # Multiply the adjacency matrix with the perturbed feature matrix for each step
            feature_matrix_for_perturbation = torch.matmul(adj_matrix, feature_matrix_for_perturbation)
            internal_diffusion.append(feature_matrix_for_perturbation.clone())

        internal_diffusion = torch.cat(internal_diffusion, dim=cat_dim)

        enriched_feature_matrices.append(internal_diffusion)

    feature_matrices_of_perturbations = torch.stack(enriched_feature_matrices)

    return feature_matrices_of_perturbations


def diffusion_sgcn(adj: Tensor, feature_matrix: Tensor, args: Any) -> Tensor:
    """
    Perform feature diffusion on the adjacency matrix for SGCN (No perturbation and no concatenation).

    Parameters:
    adj_perturbed (torch.Tensor): Tensor of adjacency matrix (n, n).
    feature_matrix (torch.Tensor): Tensor of feature matrix. (n, d)
    args (object): Arguments object with 'seed' , 'k', attributes.

    Returns:
    torch.Tensor: Tensor of enriched feature matrix after diffusion.
    """
    torch.manual_seed(args.seed)

    # Perform diffusion for 'L' steps
    for _ in range(args.L):
        feature_matrix = torch.matmul(adj, feature_matrix)

    return feature_matrix


def create_mlp(input_size: int, hidden_size: int, num_layers: int, args: Any,
               use_dropout: bool = False) -> nn.Sequential:
    """
    Create a multi-layer perceptron (MLP) with specified configuration.

    Parameters:
    input_size (int): Size of the input layer.
    hidden_size (int): Size of each hidden layer.
    num_layers (int): Number of hidden layers.
    args (Any): args object with attributes 'batch_norm', 'activation', and 'dropout'.
    use_dropout (bool): If True, include dropout layers. Defaults to False.

    Returns:
    nn.Sequential: Sequential container of the MLP layers.
    """
    layers = []
    for _ in range(num_layers):
        layers.append(nn.Linear(input_size, hidden_size))

        if args.batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))

        if args.activation == 'ELU':
            layers.append(nn.ELU())
        else:
            layers.append(nn.ReLU())

        if use_dropout:
            layers.append(nn.Dropout(args.dropout))

        input_size = hidden_size

    return nn.Sequential(*layers)


class EnrichedGraphDataset(InMemoryDataset):
    def __init__(self, root, name, dataset, p, num_perturbations, args):
        super(EnrichedGraphDataset, self).__init__(root, transform=None, pre_transform=None)
        self.name = name
        self.p = p
        self.num_perturbations = num_perturbations

        if self._processed_file_exists():
            print("Dataset was already in memory.")
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print("Preprocessing ...")
            self.data_list = self.process_dataset(dataset, args)
            self.data, self.slices = torch.load(self.processed_paths[0])

    def process_dataset(self, dataset, args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        enriched_dataset = []

        total_processing_time = 0.0

        for data in dataset:
            start_time = time.time()
            edge_index = data.edge_index
            feature_matrix = data.x.clone().float()  # Converted to float for ogb

            if args.configuration not in ["c1", "c2", "c3", "c4", "sign", "sgcn"]:
                raise ValueError("Invalid aggregation method specified")

            adj = get_adj(edge_index, set_diag=False, symmetric_normalize=args.configuration in ["sign", "sgcn"])

            # Padding if needed for OGB datasets for handling the isolated nodes
            if args.dataset in ["ogbg-molhiv", "ogbg-moltox21"]:
                if adj.size(0) != feature_matrix.size(0):
                    max_size = max(adj.size(0), feature_matrix.size(0))
                    pad_amount = max_size - adj.size(0)
                    adj = pad(adj, (0, pad_amount, 0, pad_amount), mode='constant', value=0)

            if args.configuration in ["c1", "c2", "c3", "c4"]:
                perturbed_adj = generate_perturbation(adj, self.p, self.num_perturbations, args.seed)
                normalized_adj = compute_symmetric_normalized_perturbed_adj(perturbed_adj)
                if torch.isnan(normalized_adj).any():
                    raise ValueError("NaN values encountered in normalized adjacency matrices.")
                feature_matrices_of_perts = diffusion(normalized_adj, feature_matrix, args)

                if args.configuration == 'c1':
                    final_feature_of_graph = feature_matrices_of_perts.mean(dim=0).clone()
                    final_feature_of_graph = final_feature_of_graph.sum(dim=0).unsqueeze(0).clone()

                elif args.configuration == 'c2':
                    final_feature_of_graph = feature_matrices_of_perts.mean(dim=0).clone()

                else:  # for "c3" and "c4"
                    final_feature_of_graph = feature_matrices_of_perts.view(-1, feature_matrices_of_perts.size(-1))

            elif args.configuration == "sign":
                adj = adj.unsqueeze(0).expand(1, -1, -1).clone()
                final_feature_of_graph = diffusion(adj, feature_matrix, args).squeeze()

            elif args.configuration == "sgcn":
                final_feature_of_graph = diffusion_sgcn(adj, feature_matrix, args).squeeze()

            else:
                raise ValueError("Error in choosing hyper parameters")

            end_time = time.time()
            total_processing_time += (end_time - start_time)

            enriched_data = Data(x=final_feature_of_graph, edge_index=edge_index, y=data.y)
            enriched_dataset.append(enriched_data)

        print(f"Time taken to process the dataset: {total_processing_time:.2f} seconds")
        print("Saving the dataset on the disk ...")
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


class Decoder(nn.Module):
    def __init__(self, input_size, output_size, args: Any, hidden_factor=2, batch_norm=False,
                 dropout=True):
        super(Decoder, self).__init__()

        hidden_sizes = [input_size // (hidden_factor ** i) for i in range(args.N_mlp)]
        layers = []
        for i in range(args.N_mlp - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            if args.activation == 'ELU':
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(args.dropout))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class SE2P_C1(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, args: Any):
        super(SE2P_C1, self).__init__()

        self.args = args
        self.decoder = Decoder(input_dim, output_dim, args, hidden_factor=2, batch_norm=args.batch_norm)
        # collab hidden factor=4, IMDB-M and B=3, PTC=2 and mutag=2, PROTEINS=1, ogb=2 for c1
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.decoder)

    def forward(self, data):
        x = data.x

        x = self.decoder(x)

        if self.args.dataset in ['ogbg-moltox21', 'ogbg-molhiv']:
            return x
        else:
            return F.log_softmax(x, dim=-1)


class SE2P_C2(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, args: Any):
        super(SE2P_C2, self).__init__()

        self.args = args

        self.linears = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else args.hidden_dim, args.hidden_dim)
            for i in range(args.N_pool)
        ])

        if args.batch_norm:
            self.bns = nn.ModuleList([
                nn.BatchNorm1d(args.hidden_dim)
                for _ in range(args.N_pool)
            ])

        if args.activation == 'ELU':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        if args.graph_pooling == 'sum':
            self.pool = global_add_pool
        elif args.graph_pooling == 'attention_agg':
            self.pool = AttentionalAggregation(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(args.hidden_dim, 2 * args.hidden_dim),
                                            torch.nn.BatchNorm1d(2 * args.hidden_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(2 * args.hidden_dim, 1)))

        self.decoder = Decoder(args.hidden_dim, output_dim, args)

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

        for i in range(self.args.N_pool):
            x = self.linears[i](x)
            if self.args.batch_norm:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.args.dropout, training=self.training)

        x = self.pool(x, batch)

        x = self.decoder(x)

        if self.args.dataset in ['ogbg-moltox21', 'ogbg-molhiv']:
            return x
        else:
            return F.log_softmax(x, dim=-1)


class SE2P_C3(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_perturbations: int, device, args: Any,
                 mlp_before_sum: bool = True):
        super(SE2P_C3, self).__init__()

        self.args = args
        self.mlp_before_sum = mlp_before_sum
        # MLP for individual perturbations
        self.mlp_local = create_mlp(input_size, args.hidden_dim, args.Ds_im, args)

        # MLP for aggregation
        self.mlp_global = create_mlp(args.hidden_dim, args.hidden_dim, args.Ds_om, args)

        if mlp_before_sum:
            self.mlp_before_sum = create_mlp(args.hidden_dim, args.hidden_dim, args.N_pool, args,
                                             use_dropout=True)

        if self.args.graph_pooling == 'sum':
            self.pool = global_add_pool
        elif self.args.graph_pooling == 'attention_agg':
            self.pool = AttentionalAggregation(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(args.hidden_dim, 2 * args.hidden_dim),
                                            torch.nn.BatchNorm1d(2 * args.hidden_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(2 * args.hidden_dim, 1)))

        self.decoder = Decoder(args.hidden_dim, output_size, args)

        self.num_perturbations = num_perturbations
        self.device = device

        self.reset_parameters()

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
        if self.mlp_before_sum:
            ds_output = self.mlp_before_sum(ds_output)

        batch_indexing = torch.zeros(ds_output.size(0), dtype=torch.long, device=self.device)
        start_idx = 0

        for idx, boundary in enumerate(nodes):
            batch_indexing[start_idx:start_idx + boundary] = idx
            start_idx += boundary

        x = self.pool(ds_output, batch_indexing)

        x = self.decoder(x)

        if self.args.dataset in ['ogbg-moltox21', 'ogbg-molhiv']:
            return x
        else:
            return F.log_softmax(x, dim=-1)


class SE2P_C4(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_perturbations: int, device, args: Any,
                 mlp_before_sum: bool = True):
        super(SE2P_C4, self).__init__()

        self.L = args.L
        self.args = args
        self.num_perturbations = num_perturbations
        self.device = device

        self.mlp_local_combine = create_mlp(input_size, args.hidden_dim, args.Ds_ic, args)

        self.mlp_global_combine = create_mlp(args.hidden_dim, args.hidden_dim, args.Ds_oc, args)

        self.mlp_local_merge = create_mlp(args.hidden_dim, args.hidden_dim, args.Ds_im, args)

        self.mlp_global_merge = create_mlp(args.hidden_dim, args.hidden_dim, args.Ds_om, args)

        if mlp_before_sum:
            self.mlp_before_sum = create_mlp(args.hidden_dim, args.hidden_dim, args.N_pool, args,
                                             use_dropout=True)

        if args.graph_pooling == 'sum':
            self.pool = global_add_pool
        elif args.graph_pooling == 'attention_agg':
            self.pool = AttentionalAggregation(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(args.hidden_dim, 2 * args.hidden_dim),
                                            torch.nn.BatchNorm1d(2 * args.hidden_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(2 * args.hidden_dim, 1)))

        self.decoder = Decoder(args.hidden_dim, output_size, args)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data):

        ptr = data.ptr
        nodes = (torch.diff(ptr) / (self.num_perturbations * (self.L + 1))).to(torch.long).to(self.device)

        start_comb = 0
        idx_list_all = []
        for node in nodes:
            idx_list_comb = []
            for i in range(self.num_perturbations):
                idx_comb = torch.arange(start_comb, start_comb + node).repeat(self.L + 1)
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

        if self.mlp_before_sum:
            ds_output = self.mlp_before_sum(ds_output)

        x = self.pool(ds_output, batch_indexing)

        x = self.decoder(x)

        if self.args.dataset in ['ogbg-moltox21', 'ogbg-molhiv']:
            return x
        else:
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
    args = Args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    if args.dataset in ['ogbg-molhiv', 'ogbg-moltox21']:
        raise ValueError("Invalid dataset")
    dataset = get_dataset(args)
    print(dataset)

    # Dataset statistics
    num_nodes_list, degree_list = [], []
    for graph in dataset:
        num_nodes = graph.num_nodes
        max_deg = degree(graph.edge_index[0], num_nodes, dtype=torch.long).max()
        num_nodes_list.append(num_nodes)
        degree_list.append(max_deg)

    mean_deg = torch.tensor(degree_list).float().mean()
    max_deg = torch.tensor(degree_list).max()
    min_deg = torch.tensor(degree_list).min()
    print(f'Mean Degree: {mean_deg}')
    print(f'Max Degree: {max_deg}')
    print(f'Min Degree: {min_deg}')

    mean_num_nodes = torch.tensor(num_nodes_list).float().mean().round().long().item()
    max_num_nodes = torch.tensor(num_nodes_list).float().max().round().long().item()
    min_num_nodes = torch.tensor(num_nodes_list).float().min().round().long().item()
    print(f'Mean number of nodes: {mean_num_nodes}')
    print(f'Max number of nodes: {max_num_nodes}')
    print(f'Min number of nodes: {min_num_nodes}')
    print(f'Number of graphs: {len(dataset)}')

    gamma = mean_num_nodes
    p = 2 / (1 + gamma)
    num_perturbations = gamma
    print(f'Number of perturbations: {num_perturbations}')
    print(f'Sampling probability: {p}')
    print(f'Number of features: {dataset.num_features}')
    current_path = os.getcwd()

    name = f"enriched_{args.dataset}_{args.configuration}"
    start_time = time.time()
    enriched_dataset = EnrichedGraphDataset(os.path.join(current_path, 'enriched_dataset'), name, dataset, p=p,
                                            num_perturbations=num_perturbations, args=args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Done! Time taken: {elapsed_time:.2f} seconds")
    print(f'Number of enriched features: {enriched_dataset.num_features}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f'Device: {device}')
    n_splits = 10
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    all_validation_accuracies = []
    time_seed = []

    skf_splits = separate_data(len(enriched_dataset), n_splits, args.seed)

    if args.configuration == "c1":
        model = SE2P_C1(enriched_dataset.num_features, enriched_dataset.num_classes, args).to(device)

    elif args.configuration == "c2" or args.configuration == "sign" or args.configuration == "sgcn":
        model = SE2P_C2(enriched_dataset.num_features, enriched_dataset.num_classes, args).to(device)

    elif args.configuration == "c3":
        model = SE2P_C3(enriched_dataset.num_features, enriched_dataset.num_classes, num_perturbations, device,
                        args).to(device)

    elif args.configuration == "c4":
        model = SE2P_C4(enriched_dataset.num_features, enriched_dataset.num_classes, num_perturbations, device,
                        args).to(device)
    else:
        raise ValueError("Error in choosing the model.")

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
                                              len(train_indices.tolist()) / args.batch_size)),
                                  generator=generator),
            batch_size=args.batch_size, drop_last=False,
            collate_fn=Collater(follow_batch=[], exclude_keys=[]))

        test_loader = DataLoader(enriched_dataset[test_indices.tolist()], batch_size=args.batch_size,
                                 shuffle=False)

        if fold == 0:
            print(f'Model learnable parameters for {model.__class__.__name__}: {count_parameters(model)}')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
    average_validation_curve = torch.stack(all_validation_accuracies, dim=0)
    acc_mean = average_validation_curve.mean(dim=0)
    best_epoch = acc_mean.argmax().item()
    best_epoch_mean = average_validation_curve[:, best_epoch].mean()
    std_at_max_avg_validation_acc_epoch = average_validation_curve[:, best_epoch].std()

    print(f'Epoch {best_epoch + 1} got maximum averaged validation accuracy in seed {args.seed}: '
          f'{best_epoch_mean}')
    print(f'Standard Deviation for the results of epoch {best_epoch + 1} over all the folds '
          f'in seed {args.seed}: {std_at_max_avg_validation_acc_epoch}')
    print(f'Average total time taken for each fold in seed {args.seed}: {np.mean(time_seed)}')
    print(f'STD total time taken for each fold in seed {args.seed}: {np.std(time_seed)}')
    print(f'Average Time/Epoch in seed {args.seed}: {np.mean(time_per_epoch)}')
    print(f'STD Time/Epoch in seed {args.seed}: {np.std(time_per_epoch)}')


if __name__ == "__main__":
    main()
