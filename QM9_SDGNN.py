import os
import os.path as osp
import argparse
import time
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import Set2Set
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import remove_self_loops, degree
from torch_geometric.data import Data, Dataset
import wandb

sweep_config = {
    "method": "bayes",
    "metric": {"name": "test_error", "goal": "minimize"},
    "parameters": {
        "normalization": {"values": ["Before", "After"]},
        "k": {"values": [2, 3, 4]},
        "sum_or_cat": {"values": ["cat", "sum"]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="QM9")


class MyTransform(object):
    def __call__(self, data, args):
        # Specify target.
        data.y = data.y[:, args.target]
        return data


def get_adj(edge_index):
    adj = to_dense_adj(edge_index).squeeze()
    identity_matrix = torch.eye(adj.shape[0])
    adj_with_self_loops = adj + identity_matrix
    return adj_with_self_loops


def compute_symmetric_normalized_adj(edge_index):
    #  Convert to dense adjacency matrix
    adj = to_dense_adj(edge_index).squeeze()
    identity_matrix = torch.eye(adj.shape[0])
    adj_with_self_loops = adj + identity_matrix
    # Compute the degree matrix (D) by summing over rows
    D = torch.diag(adj_with_self_loops.sum(dim=1))

    # Compute the inverse square root of the degree matrix (D_inv_sqrt)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diagonal()))

    # Apply symmetric normalization to the adjacency matrix
    adj_normalized = torch.mm(torch.mm(D_inv_sqrt, adj_with_self_loops), D_inv_sqrt)

    return adj_normalized


def generate_perturbation(adj, p):
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
    # Convert to dense adjacency matrix
    # adj = to_dense_adj(edge_index).squeeze()
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


def diffusion(adj_perturbed, feature_matrix, config):
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
        else:
            internal_diffusion = torch.cat(internal_diffusion, dim=0)

        enriched_feature_matrices.append(internal_diffusion)

    feature_matrices_of_perturbations = torch.stack(enriched_feature_matrices)

    return feature_matrices_of_perturbations


class EnrichedGraphDataset(Dataset):
    def __init__(self, root, dataset, p, num_perturbations, max_nodes, config, args):
        super(EnrichedGraphDataset, self).__init__(root, transform=None, pre_transform=None)
        # self.k = k
        self.p = p
        self.num_perturbations = num_perturbations
        self.max_nodes = max_nodes
        self.data_list = self.process_dataset(dataset, config, args)

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
        # dataset = TUDataset(self.root, name)
        enriched_dataset = []
        final_feature_of_graph = None

        for data in dataset:
            edge_index = data.edge_index
            feature_matrix = data.x.clone()

            if args.agg == "deepset" and config.normalization == "Before":
                adj = compute_symmetric_normalized_adj(edge_index)
                feature_matrix, normalized_adj = self.pad_data(feature_matrix, adj)
                normalized_adj_1 = normalized_adj.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()
                perturbed_adj = generate_perturbation(normalized_adj_1, self.p)
                feature_matrices_of_perts = diffusion(perturbed_adj, feature_matrix, config)
                final_feature_of_graph = feature_matrices_of_perts

            elif args.agg == "deepset" and config.normalization == "After":
                adj = get_adj(edge_index)
                feature_matrix, adjacency = self.pad_data(feature_matrix, adj)
                adj = adjacency.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()
                perturbed_adj = generate_perturbation(adj, self.p)
                normalized_adj = compute_symmetric_normalized_perturbed_adj(perturbed_adj)
                if torch.isnan(normalized_adj).any():
                    raise ValueError("NaN values encountered in normalized adjacency matrices.")
                feature_matrices_of_perts = diffusion(normalized_adj, feature_matrix, config)
                final_feature_of_graph = feature_matrices_of_perts

            elif args.agg == "mean" and config.normalization == "Before":
                normalized_adj = compute_symmetric_normalized_adj(edge_index)
                normalized_adj_1 = normalized_adj.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()
                perturbed_adj = generate_perturbation(normalized_adj_1, self.p)
                feature_matrices_of_perts = diffusion(perturbed_adj, feature_matrix, config)
                final_feature_of_graph = feature_matrices_of_perts.mean(dim=0).clone()

            elif args.agg == "mean" and config.normalization == "After":
                adjacency = get_adj(edge_index)
                adj = adjacency.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()
                perturbed_adj = generate_perturbation(adj, self.p)
                normalized_adj = compute_symmetric_normalized_perturbed_adj(perturbed_adj)
                if torch.isnan(normalized_adj).any():
                    raise ValueError("NaN values encountered in normalized adjacency matrices.")
                feature_matrices_of_perts = diffusion(normalized_adj, feature_matrix, config)
                final_feature_of_graph = feature_matrices_of_perts.mean(dim=0).clone()

            elif args.agg == "concat" and config.normalization == "Before":
                normalized_adj = compute_symmetric_normalized_adj(edge_index)
                normalized_adj_1 = normalized_adj.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()
                perturbed_adj = generate_perturbation(normalized_adj_1, self.p)
                feature_matrices_of_perts = diffusion(perturbed_adj, feature_matrix, config)
                final_feature_of_graph = feature_matrices_of_perts.view(-1, feature_matrices_of_perts.size(-1)).clone()

            elif args.agg == "concat" and config.normalization == "After":
                adjacency = get_adj(edge_index)
                adj = adjacency.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()
                perturbed_adj = generate_perturbation(adj, self.p)
                normalized_adj = compute_symmetric_normalized_perturbed_adj(perturbed_adj)
                if torch.isnan(normalized_adj).any():
                    raise ValueError("NaN values encountered in normalized adjacency matrices.")
                feature_matrices_of_perts = diffusion(normalized_adj, feature_matrix, config)
                final_feature_of_graph = feature_matrices_of_perts.view(-1, feature_matrices_of_perts.size(-1)).clone()

            enriched_data = Data(x=final_feature_of_graph, edge_index=edge_index, edge_attr=data.edge_attr, y=data.y,
                                 pos=data.pos, idx=data.idx, name=data.name, z=data.z)
            enriched_dataset.append(enriched_data)

        return enriched_dataset

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        M_in, M_out = dataset.num_features, 32
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
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--agg', type=str, default="deepset", choices=["mean", "concat", "deepset"],
                        help='Method for aggregating the perturbation')
    args = parser.parse_args()
    print(args)
    target = int(args.target)
    print('---- Target: {} ----'.format(target))

    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'MPNN-QM9')
    transform = T.Compose([MyTransform(), T.Distance()])
    dataset = QM9(path, transform=transform).shuffle()

    n = []
    degs = []
    for g in dataset:
        num_nodes = g.num_nodes
        deg = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
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
        start_time = time.time()
        print("Preprocessing ...")
        enriched_dataset = EnrichedGraphDataset(os.path.join(os.getcwd(), 'enriched_dataset'), dataset, p=p,
                                                num_perturbations=num_runs, max_nodes=max_nodes, args=args,
                                                config=config)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")

        tenpercent = int(len(enriched_dataset) * 0.1)
        mean = torch.stack([data.y for data in enriched_dataset[tenpercent:]]).mean(dim=0)
        std = torch.stack([data.y for data in enriched_dataset[tenpercent:]]).std(dim=0)

        for data in enriched_dataset:
            data.y = (data.y - mean) / std

        test_dataset = enriched_dataset[:tenpercent]
        val_dataset = enriched_dataset[tenpercent:2 * tenpercent]
        train_dataset = enriched_dataset[2 * tenpercent:]
        test_loader = DataLoader(test_dataset, batch_size=64)
        val_loader = DataLoader(val_dataset, batch_size=64)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net(enriched_dataset).to(device)
        print(f"Number of trainable parameters: {count_parameters(model)}")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.7, patience=5,
                                                               min_lr=0.00001)
        # mean, std = mean[target].to(device), std[target].to(device)

        print(model.__class__.__name__)
        best_val_error = None
        for epoch in range(1, 301):
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
            print('Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
                  'Test MAE: {:.7f}, Time: {:.2f} seconds'.format(epoch, lr, loss, val_error, test_error, epoch_time),
                  flush=True)
            wandb.log({
                "loss": loss,
                "val_error": val_loader,
                "test_error": test_error
            })


if __name__ == "__main__":
    wandb.agent(sweep_id, main, count=5)
