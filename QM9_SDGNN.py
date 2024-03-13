import os
import os.path as osp
import argparse
import time
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
import torch.nn as nn
from torch_geometric.nn import Set2Set
from torch.nn.functional import pad
from torch_geometric.nn import NNConv
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import remove_self_loops, degree
from torch_geometric.data import Data, Dataset, InMemoryDataset
import numpy as np
import wandb

sweep_config = {
    "method": "grid",
    "metric": {"name": "test_error", "goal": "minimize"},
    "parameters": {
        "normalization": {"values": ["After"]},
        "k": {"values": [2]},
        "sum_or_cat": {"values": ["cat"]},
        "hidden_dim": {"values": [16, 32, 64]},
        "batch_size": {"values": [32, 64, 128]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="SDGNN_QM9")


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


def generate_perturbation(adj, p, seed):
    torch.manual_seed(seed)

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


def diffusion(adj_perturbed, feature_matrix, config, seed):
    torch.manual_seed(seed)
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
            internal_diffusion = torch.cat(internal_diffusion, dim=1)
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
    def __init__(self, root, name, dataset, p, num_perturbations, max_nodes, config, args):
        super(EnrichedGraphDataset, self).__init__(root, transform=None, pre_transform=None)
        self.name = name
        self.p = p
        self.num_perturbations = num_perturbations
        self.max_nodes = max_nodes

        if self._processed_file_exists():
            print("Dataset was already in memory.")
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print("Preprocessing ...")
            self.data_list = self.process_dataset(dataset, config, args)
            self.data, self.slices = torch.load(self.processed_paths[0])

    def pad_data(self, feature_matrix, adj):
        num_nodes = feature_matrix.size(0)

        # Pad feature matrix if necessary
        if num_nodes < self.max_nodes:
            pad_size = self.max_nodes - num_nodes
            feature_matrix = torch.cat([feature_matrix, torch.zeros(pad_size, feature_matrix.size(1))], dim=0)

        # Pad adjacency if necessary
        if adj.size(0) < self.max_nodes:
            pad_size = self.max_nodes - adj.size(0)
            zeros_pad = torch.zeros(pad_size, adj.size(1))
            adj = torch.cat([adj, zeros_pad], dim=0)
            adj = torch.cat([adj, torch.zeros(adj.size(0), pad_size)], dim=1)

        return feature_matrix, adj

    def process_dataset(self, dataset, config, args):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        enriched_dataset = []
        # final_feature_of_graph = None
        counter = 0
        for data in dataset:
            edge_index = data.edge_index
            feature_matrix = data.x.clone().float()  # Converted to float for ogb

            if args.agg not in ["deepset", "mean", "concat", "sign", "sgcn"]:
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
                adj = adj.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()  # TODO move this to the  function
                perturbed_adj = generate_perturbation(adj, self.p, args.seed)
                feature_matrices_of_perts = diffusion(perturbed_adj, feature_matrix, config, args.seed)
                final_feature_of_graph = feature_matrices_of_perts.view(-1, feature_matrices_of_perts.size(
                    -1))  # remove view if wanted to use previous

            elif args.agg == "deepset" and config.normalization == "After":
                adj = get_adj(edge_index, set_diag=False,
                              symmetric_normalize=False)  # set diag here can be false or true
                feature_matrix, adj = self.pad_data(feature_matrix, adj)
                adj = adj.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()
                perturbed_adj = generate_perturbation(adj, self.p, args.seed)
                normalized_adj = compute_symmetric_normalized_perturbed_adj(perturbed_adj)
                if torch.isnan(normalized_adj).any():
                    raise ValueError("NaN values encountered in normalized adjacency matrices.")
                feature_matrices_of_perts = diffusion(normalized_adj, feature_matrix, config, args.seed)
                final_feature_of_graph = feature_matrices_of_perts.view(-1, feature_matrices_of_perts.size(-1))

            elif args.agg == "mean" and config.normalization == "Before":
                adj = get_adj(edge_index, set_diag=False, symmetric_normalize=True)
                if adj.size(0) != feature_matrix.size(0):
                    counter = counter + 1
                    max_size = max(adj.size(0), feature_matrix.size(0))
                    pad_amount = max_size - adj.size(0)
                    adj = pad(adj, (0, pad_amount, 0, pad_amount), mode='constant', value=0)
                adj = adj.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()
                perturbed_adj = generate_perturbation(adj, self.p, args.seed)
                feature_matrices_of_perts = diffusion(perturbed_adj, feature_matrix, config, args.seed)
                final_feature_of_graph = feature_matrices_of_perts.mean(dim=0).clone()

            elif args.agg == "mean" and config.normalization == "After":
                adj = get_adj(edge_index, set_diag=False, symmetric_normalize=False)
                if adj.size(0) != feature_matrix.size(0):
                    counter = counter + 1
                    max_size = max(adj.size(0), feature_matrix.size(0))
                    pad_amount = max_size - adj.size(0)
                    adj = pad(adj, (0, pad_amount, 0, pad_amount), mode='constant', value=0)
                adj = adj.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()
                perturbed_adj = generate_perturbation(adj, self.p, args.seed)
                normalized_adj = compute_symmetric_normalized_perturbed_adj(perturbed_adj)
                if torch.isnan(normalized_adj).any():
                    raise ValueError("NaN values encountered in normalized adjacency matrices.")
                feature_matrices_of_perts = diffusion(normalized_adj, feature_matrix, config, args.seed)
                final_feature_of_graph = feature_matrices_of_perts.mean(dim=0).clone()

            elif args.agg == "concat" and config.normalization == "Before":
                adj = get_adj(edge_index, set_diag=True, symmetric_normalize=True)
                adj = adj.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()
                perturbed_adj = generate_perturbation(adj, self.p, args.seed)
                feature_matrices_of_perts = diffusion(perturbed_adj, feature_matrix, config, args.seed)
                final_feature_of_graph = feature_matrices_of_perts.view(-1, feature_matrices_of_perts.size(-1)).clone()

            elif args.agg == "concat" and config.normalization == "After":
                adj = get_adj(edge_index, set_diag=False, symmetric_normalize=False)
                adj = adj.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()
                perturbed_adj = generate_perturbation(adj, self.p, args.seed)
                normalized_adj = compute_symmetric_normalized_perturbed_adj(perturbed_adj)
                if torch.isnan(normalized_adj).any():
                    raise ValueError("NaN values encountered in normalized adjacency matrices.")
                feature_matrices_of_perts = diffusion(normalized_adj, feature_matrix, config, args.seed)
                final_feature_of_graph = feature_matrices_of_perts.view(-1, feature_matrices_of_perts.size(-1)).clone()

            elif args.agg == "sign":
                adj = get_adj(edge_index, set_diag=False, symmetric_normalize=True)
                adj = adj.unsqueeze(0).expand(1, -1, -1).clone()
                final_feature_of_graph = diffusion(adj, feature_matrix, config, args.seed)
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


class DeepSet(nn.Module):
    def __init__(self, input_size, hidden_size, num_perturbations, max_nodes):
        super(DeepSet, self).__init__()

        self.hidden_size = hidden_size
        # MLP for individual perturbations
        self.mlp_perturbation = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU()
        )

        # MLP for aggregation
        self.mlp_aggregation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU()
        )

        self.num_perturbations = num_perturbations
        self.max_nodes = max_nodes

    def forward(self, input_data):
        # batch_size = input_data.size(0)

        x = self.mlp_perturbation(input_data)
        # x_transformed = x.view(self.num_perturbations, input_data.size(0) // self.num_perturbations,
        # self.hidden_size)  for batch=1

        # x_transformed = x.view(batch_size // self.num_perturbations, self.num_perturbations, self.hidden_size)
        x_transformed = x.view(-1, self.num_perturbations, self.max_nodes, self.hidden_size)
        aggregated_output = torch.sum(x_transformed, dim=1)
        aggregated_output = aggregated_output.view(-1, self.hidden_size)
        # aggregated_output = torch.sum(x_transformed, dim=0) va in bara batch=1 bod

        final_output = self.mlp_aggregation(aggregated_output)

        return final_output


class Net_deepset(torch.nn.Module):
    def __init__(self, enriched_dataset, hidden_dim, num_perturbations, max_nodes):
        super(Net_deepset, self).__init__()

        self.num_perturbations = num_perturbations

        self.deepset_aggregator = DeepSet(enriched_dataset.num_features, hidden_dim, num_perturbations, max_nodes)
        M_in, M_out = hidden_dim, 32
        self.conv1 = Sequential(Linear(M_in, 128), ReLU(), Linear(128, M_out))

        M_in, M_out = M_out, 64
        self.conv2 = Sequential(Linear(M_in, 128), ReLU(), Linear(128, M_out))

        M_in, M_out = M_out, 64
        self.conv3 = Sequential(Linear(M_in, 128), ReLU(), Linear(128, M_out))

        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):
        x = data.x
        batch = data.batch

        aggregated_features = self.deepset_aggregator(x)
        batch = batch.view(-1, self.num_perturbations).to(torch.float).mean(dim=1).long()

        x = F.elu(self.conv1(aggregated_features))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))

        x = scatter_mean(x, batch, dim=0)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


class Net(torch.nn.Module):
    def __init__(self, enriched_dataset):
        super(Net, self).__init__()
        M_in, M_out = enriched_dataset.num_features, 32
        self.conv1 = Sequential(Linear(M_in, 128), ReLU(), Linear(128, M_out))

        M_in, M_out = M_out, 64
        self.conv2 = Sequential(Linear(M_in, 128), ReLU(), Linear(128, M_out))

        M_in, M_out = M_out, 64
        self.conv3 = Sequential(Linear(M_in, 128), ReLU(), Linear(128, M_out))

        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)

    def forward(self, data):
        x = data.x
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))

        x = scatter_mean(x, data.batch, dim=0)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1)


def train(model, loader, device, optimizer):
    model.train()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = F.mse_loss(pred, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(loader.dataset)


def test(model, loader, device, std):
    model.eval()
    error = 0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        error += ((pred * std) -
                  (data.y * std)).abs().sum().item()  # MAE
    return error / len(loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(config=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default=0)
    # parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--agg', type=str, default="deepset", choices=["mean", "concat", "deepset", "sign", "sgcn"],
                        help='Method for aggregating the perturbation')
    args = parser.parse_args()
    print(args)
    target = int(args.target)
    print('---- Target: {} ----'.format(target))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    class MyTransform(object):
        def __call__(self, data):
            # Specify target.
            data.y = data.y[:, args.target]
            return data

    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'MPNN-QM9')
    transform = T.Compose([MyTransform(), T.Distance()])
    dataset = QM9(path, transform=transform).shuffle()

    n = []
    degs = []
    for g in dataset:
        num_nodes = g.num_nodes
        deg = degree(g.edge_index[0], num_nodes, dtype=torch.long)
        n.append(g.num_nodes)
        degs.append(deg.max())
    print(f'Mean Degree: {torch.stack(degs).float().mean()}')
    print(f'Max Degree: {torch.stack(degs).max()}')
    print(f'Min Degree: {torch.stack(degs).min()}')
    mean_n = torch.tensor(n).float().mean().round().long().item()
    print(f'Mean number of nodes: {mean_n}')
    max_nodes = torch.tensor(n).float().max().round().long().item()
    print(f'Max number of nodes: {max_nodes}')
    print(f'Min number of nodes: {torch.tensor(n).float().min().round().long().item()}')
    print(f'Number of graphs: {len(dataset)}')
    gamma = mean_n
    p = 2 * 1 / (1 + gamma)
    num_runs = gamma
    print(f'Number of runs: {num_runs}')
    print(f'Sampling probability: {p}')
    print("====================================")

    with wandb.init(config=config):
        config = wandb.config
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        if args.agg == "mean":
            name = "enriched_qm9_mean"
        else:
            name = "enriched_qm9_ds"

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        start_time = time.time()
        # print("Preprocessing ...")
        enriched_dataset = EnrichedGraphDataset(os.path.join(os.getcwd(), 'enriched_dataset'), name, dataset, p=p,
                                                num_perturbations=num_runs, max_nodes=max_nodes, args=args,
                                                config=config)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")

        tenpercent = int(len(enriched_dataset) * 0.1)
        mean = torch.stack([data.y for data in enriched_dataset[tenpercent:]]).mean(dim=0)
        std = torch.stack([data.y for data in enriched_dataset[tenpercent:]]).std(dim=0)
        mean = mean.to(device)
        std = std.to(device)

        for data in enriched_dataset:
            data.y = data.y.to(device)
            data.y = (data.y - mean) / std

        test_dataset = enriched_dataset[:tenpercent]
        val_dataset = enriched_dataset[tenpercent:2 * tenpercent]
        train_dataset = enriched_dataset[2 * tenpercent:]
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        if args.agg == 'mean':
            model = Net(enriched_dataset).to(device)
        else:
            print("deepset model selected")
            model = Net_deepset(enriched_dataset, config.hidden_dim, num_perturbations=num_runs, max_nodes=max_nodes).to(
                device)

        print(f"Number of trainable parameters: {count_parameters(model)}")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.7, patience=5,
                                                               min_lr=0.00001)
        # mean, std = mean[target].to(device), std[target].to(device)

        print(model.__class__.__name__)
        best_val_error = None
        time_per_epoch = []
        for epoch in range(1, 5):
            start_time = time.time()
            lr = scheduler.optimizer.param_groups[0]['lr']
            loss = train(model, train_loader, device, optimizer)
            val_error = test(model, val_loader, device, std)
            scheduler.step(val_error)

            if best_val_error is None or val_error <= best_val_error:
                test_error = test(model, test_loader, device, std)
                best_val_error = val_error

            end_time = time.time()
            epoch_time = end_time - start_time
            time_per_epoch.append(epoch_time)
            print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
                  'Test MAE: {:.7f}, Time: {:.2f} seconds'.format(epoch, lr, loss, val_error, test_error, epoch_time),
                  flush=True)
            wandb.log({
                "loss": loss,
                "val_error": val_error,
                "test_error": test_error
            })
            if epoch == 150:
                print(np.mean(time_per_epoch))
                print(np.std(time_per_epoch))


if __name__ == "__main__":
    wandb.agent(sweep_id, main)
