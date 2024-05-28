import torch
import os
from args import Args
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
from torch.nn.functional import pad
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader.dataloader import Collater
import time
from torch_geometric.data import Data, InMemoryDataset
import wandb
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.inits import reset
from datasets import get_dataset

sweep_config = {
    "method": "grid",
    "metric": {"name": "test_acc", "goal": "maximize"},
    "parameters": {
        "lr": {"values": [0.01]},
        "num_layers": {"values": [2, 3, 4]},
        "batch_norm": {"values": [True]},
        "batch_size": {"values": [32]},
        "dropout": {"values": [0.2, 0.5]},  # if used for c2, use for c3 and others too.
        "k": {"values": [3]},
        # TODO specify for which was 2 and which was 3. pro(3 or 2?) ptc(3) imdbm & b (2) collab (2) mutag (3)
        "sum_or_cat": {"values": ["cat"]},
        "decoder_layers": {"values": [2]},
        "activation": {"values": ["ReLU"]},
        "ds_local_layers_comb": {"values": [2]},
        "ds_global_layers_comb": {"values": [2]},
        "ds_local_layers_merge": {"values": [1, 2]},
        "ds_global_layers_merge": {"values": [1, 2]},
        "hidden_dim": {"values": [32]},
        "graph_pooling": {"values": ["sum"]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="test")


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

        if config.sum_or_cat == "sum":  # sum is not used in our model
            internal_diffusion = torch.stack(internal_diffusion, dim=0)
            internal_diffusion = torch.sum(internal_diffusion, dim=0)
        elif config.sum_or_cat == "cat":
            internal_diffusion = torch.cat(internal_diffusion, dim=0 if args.configuration == "c4" else 1)
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


def create_mlp(input_size, hidden_size, num_layers, config, use_dropout=False):
    layers = []
    for _ in range(num_layers):
        layers.append(nn.Linear(input_size, hidden_size))
        if config.batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))
        if config.activation == 'ELU':
            layers.append(nn.ELU())
        else:
            layers.append(nn.ReLU())
        if use_dropout:
            layers.append(nn.Dropout(config.dropout))
        input_size = hidden_size
    return nn.Sequential(*layers)


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

    def process_dataset(self, dataset, config, args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        enriched_dataset = []
        # counter = 0
        for data in dataset:
            edge_index = data.edge_index
            feature_matrix = data.x.clone().float()  # Converted to float for ogb

            if args.configuration not in ["c1", "c2", "c3", "c4", "sign", "sgcn"]:
                raise ValueError("Invalid aggregation method specified")

            adj = get_adj(edge_index, set_diag=False, symmetric_normalize=args.configuration in ["sign", "sgcn"])

            # Handle padding if needed for OGB datasets
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
                feature_matrices_of_perts = diffusion(normalized_adj, feature_matrix, config, args)

                if args.configuration in ["c1", "c2"]:
                    final_feature_of_graph = feature_matrices_of_perts.mean(dim=0).clone()
                else:  # for "c3" and "c4"
                    final_feature_of_graph = feature_matrices_of_perts.view(-1, feature_matrices_of_perts.size(-1))

            elif args.configuration == "sign":
                adj = adj.unsqueeze(0).expand(1, -1, -1).clone()
                final_feature_of_graph = diffusion(adj, feature_matrix, config, args).squeeze()

            elif args.configuration == "sgcn":
                adj = adj.unsqueeze(0).expand(1, -1, -1).clone()
                final_feature_of_graph = diffusion_sgcn(adj, feature_matrix, config, args.seed).squeeze()

            else:
                raise ValueError("Error in choosing hyper parameters")

            enriched_data = Data(x=final_feature_of_graph, edge_index=edge_index, y=data.y)
            enriched_dataset.append(enriched_data)

        # print(counter)
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


class Decoder(nn.Module):  # This is for the decoder of the SDGNN C2, C3 and C4.
    def __init__(self, input_size, output_size, config, hidden_factor=2, batch_norm=False,
                 dropout=True):  # collab hidden factor=4, IMDB-M and B=3, PTC=2 and mutag=2, PROTEINS=1 for c1
        super(Decoder, self).__init__()

        hidden_sizes = [input_size // (hidden_factor ** i) for i in range(config.decoder_layers)]
        layers = []
        for i in range(config.decoder_layers - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            if config.activation == 'ELU':
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class SE2P_C1(nn.Module):
    def __init__(self, input_dim, output_dim, config, args):
        super(SE2P_C1, self).__init__()

        self.args = args
        self.decoder = Decoder(input_dim, output_dim, config, hidden_factor=4, batch_norm=config.batch_norm)

        # self.decoder = MLP(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=output_dim,
        #                    num_layers=decoder_layers, batch_norm="batch_norm" if config.batch_norm else None,
        #                    dropout=[dropout] * decoder_layers, activation=config.activation)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.decoder)

    def forward(self, data):
        x = data.x
        batch = data.batch

        x = global_add_pool(x, batch)

        x = self.decoder(x)

        if self.args.dataset in ['ogbg-moltox21', 'ogbg-molhiv']:
            return x
        else:
            return F.log_softmax(x, dim=-1)


class SE2P_C2(nn.Module):
    def __init__(self, input_dim, output_dim, config, args):
        super(SE2P_C2, self).__init__()

        self.config = config
        self.args = args

        self.linears = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else config.hidden_dim, config.hidden_dim)
            for i in range(config.num_layers)
        ])

        if config.batch_norm:
            self.bns = nn.ModuleList([
                nn.BatchNorm1d(config.hidden_dim)
                for _ in range(config.num_layers)
            ])

        if config.activation == 'ELU':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

        if config.graph_pooling == 'sum':
            self.pool = global_add_pool
        elif self.config.graph_pooling == 'attention_agg':
            self.pool = AttentionalAggregation(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(config.hidden_dim, 2 * config.hidden_dim),
                                            torch.nn.BatchNorm1d(2 * config.hidden_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(2 * config.hidden_dim, 1)))

        self.decoder = Decoder(config.hidden_dim, output_dim, config)

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

        for i in range(self.config.num_layers):
            x = self.linears[i](x)
            if self.config.batch_norm:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.config.dropout, training=self.training)

        x = self.pool(x, batch)

        x = self.decoder(x)

        if self.args.dataset in ['ogbg-moltox21', 'ogbg-molhiv']:
            return x
        else:
            return F.log_softmax(x, dim=-1)


class SE2P_C3(nn.Module):
    def __init__(self, input_size, output_size, num_perturbations, device, config, args, mlp_before_sum=True):
        super(SE2P_C3, self).__init__()

        self.config = config
        self.args = args
        self.mlp_before_sum = mlp_before_sum
        # MLP for individual perturbations
        self.mlp_local = create_mlp(input_size, config.hidden_dim, config.ds_local_layers_merge, config)

        # MLP for aggregation
        self.mlp_global = create_mlp(config.hidden_dim, config.hidden_dim, config.ds_global_layers_merge, config)

        if mlp_before_sum:
            self.mlp_before_sum = create_mlp(config.hidden_dim, config.hidden_dim, config.num_layers, config,
                                             use_dropout=True)

        if self.config.graph_pooling == 'sum':
            self.pool = global_add_pool
        elif self.config.graph_pooling == 'attention_agg':
            self.pool = AttentionalAggregation(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(config.hidden_dim, 2 * config.hidden_dim),
                                            torch.nn.BatchNorm1d(2 * config.hidden_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(2 * config.hidden_dim, 1)))

        self.decoder = Decoder(config.hidden_dim, output_size, config)

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
    def __init__(self, input_size, output_size, num_perturbations, device, config, args, mlp_before_sum=True):
        super(SE2P_C4, self).__init__()

        self.k = config.k
        self.args = args
        self.num_perturbations = num_perturbations
        self.device = device

        self.mlp_local_combine = create_mlp(input_size, config.hidden_dim, config.ds_local_layers_comb, config)

        self.mlp_global_combine = create_mlp(config.hidden_dim, config.hidden_dim, config.ds_global_layers_comb,
                                             config)

        self.mlp_local_merge = create_mlp(config.hidden_dim, config.hidden_dim, config.ds_local_layers_merge, config)

        self.mlp_global_merge = create_mlp(config.hidden_dim, config.hidden_dim, config.ds_global_layers_merge,
                                           config)

        if mlp_before_sum:
            self.mlp_before_sum = create_mlp(config.hidden_dim, config.hidden_dim, config.num_layers, config,
                                             use_dropout=True)

        if config.graph_pooling == 'sum':
            self.pool = global_add_pool
        elif config.graph_pooling == 'attention_agg':
            self.pool = AttentionalAggregation(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(config.hidden_dim, 2 * config.hidden_dim),
                                            torch.nn.BatchNorm1d(2 * config.hidden_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(2 * config.hidden_dim, 1)))

        self.decoder = Decoder(config.hidden_dim, output_size, config)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

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
    print(f'Number of features: {dataset.num_features}')
    current_path = os.getcwd()

    wandb.login()
    with wandb.init(config=config):
        config = wandb.config
        # print(args)
        name = f"enriched_{args.dataset}_{args.configuration}"
        start_time = time.time()
        enriched_dataset = EnrichedGraphDataset(os.path.join(current_path, 'enriched_dataset'), name, dataset, p=p,
                                                num_perturbations=num_perturbations, config=config, args=args)

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
            model = SE2P_C1(enriched_dataset.num_features, enriched_dataset.num_classes, config, args).to(device)

        elif args.configuration == "c2" or args.configuration == "sign" or args.configuration == "sgcn":
            model = SE2P_C2(enriched_dataset.num_features, enriched_dataset.num_classes, config, args).to(device)

        elif args.configuration == "c3":
            model = SE2P_C3(enriched_dataset.num_features, enriched_dataset.num_classes, num_perturbations, device,
                             config, args).to(device)

        elif args.configuration == "c4":
            model = SE2P_C4(enriched_dataset.num_features, enriched_dataset.num_classes, num_perturbations, device,
                             config, args).to(device)
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

        wandb.log(
            {"Test Accuracy": best_epoch_mean,
             "Std": std_at_max_avg_validation_acc_epoch
             })


if __name__ == "__main__":
    wandb.agent(sweep_id, main)
