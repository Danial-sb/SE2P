import torch
import os
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import RandomSampler
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter
from torch_geometric.utils import degree
from sklearn.model_selection import StratifiedKFold, KFold
from torch_geometric.loader.dataloader import Collater
import time
import argparse
import os.path as osp
from torch_geometric.data import Data
import wandb
from ptc_dataset import PTCDataset

sweep_config = {
    "method": "random",
    "metric": {"name": "final_acc", "goal": "maximize"},
    "parameters": {
        "lr": {"values": [0.001, 0.01]},
        "batch_size": {"values": [64, 32]},
        "dropout": {"values": [0.1, 0.2, 0.5]},
        "normalization": {"values": ["Before", "After"]},
        "hidden_dim": {"values": [16, 32, 64, 128]}
    }
}
sweep_id = wandb.sweep(sweep_config, project="SDGNN")


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
    else:
        raise ValueError("Invalid dataset name")

    return dataset


def separate_data(dataset_len, seed):
    # Use same splitting/10-fold as GIN paper
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    idx_list = []
    for idx in skf.split(np.zeros(dataset_len), np.zeros(dataset_len)):
        idx_list.append(idx)
    return idx_list


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


def diffusion(adj_perturbed, feature_matrix, k):  # change this for algorithm 1
    enriched_feature_matrices = []
    for perturbation in range(adj_perturbed.size(0)):
        # Get the adjacency matrix for this perturbation
        adj_matrix = adj_perturbed[perturbation]
        feature_matrix_for_perturbation = feature_matrix.clone()

        internal_diffusion = [feature_matrix_for_perturbation.clone()]
        # Perform diffusion for 'k' steps
        for _ in range(k):
            # Multiply the adjacency matrix with the perturbed feature matrix for each step
            feature_matrix_for_perturbation = torch.matmul(adj_matrix, feature_matrix_for_perturbation)
            internal_diffusion.append(feature_matrix_for_perturbation.clone())
        # The following two lines are for using sum in eq 1
        # internal_diffusion = torch.stack(internal_diffusion, dim=0)
        # internal_diffusion = torch.sum(internal_diffusion, dim=0)
        internal_diffusion = torch.cat(internal_diffusion, dim=0)  # This is eq 1 when cat is used
        enriched_feature_matrices.append(internal_diffusion)

    feature_matrices_of_perturbations = torch.stack(enriched_feature_matrices)

    return feature_matrices_of_perturbations


class EnrichedGraphDataset(Dataset):
    def __init__(self, root, dataset, k, p, num_perturbations, config, args):
        super(EnrichedGraphDataset, self).__init__(root, transform=None, pre_transform=None)
        self.k = k
        self.p = p
        self.num_perturbations = num_perturbations
        self.data_list = self.process_dataset(dataset, config, args)

    def process_dataset(self, dataset, config, args):
        # dataset = TUDataset(self.root, name)
        enriched_dataset = []
        feature_matrices_of_perts = None
        final_feature_of_graph = None

        for data in dataset:
            edge_index = data.edge_index
            feature_matrix = data.x.clone()

            if config.normalization == "Before":
                normalized_adj = compute_symmetric_normalized_adj(edge_index)
                perturbed_adj = generate_perturbation(normalized_adj, self.p)
                feature_matrices_of_perts = diffusion(perturbed_adj, feature_matrix, self.k)

            elif config.normalization == "After":
                adjacency = get_adj(edge_index)
                adj = adjacency.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()
                perturbed_adj = generate_perturbation(adj, self.p)
                normalized_adj = compute_symmetric_normalized_perturbed_adj(perturbed_adj)
                if torch.isnan(normalized_adj).any():
                    raise ValueError("NaN values encountered in normalized adjacency matrices.")
                feature_matrices_of_perts = diffusion(normalized_adj, feature_matrix, self.k)

            if args.agg == "mean":
                final_feature_of_graph = feature_matrices_of_perts.mean(dim=0).clone()

            elif args.agg == "concat":
                final_feature_of_graph = feature_matrices_of_perts.view(-1, feature_matrices_of_perts.size(-1)).clone()

            if final_feature_of_graph is not None:
                enriched_data = Data(x=final_feature_of_graph, edge_index=edge_index, y=data.y)
                enriched_dataset.append(enriched_data)
            else:
                raise ValueError("No aggregation method specified.")

        return enriched_dataset

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


class DropGNN_V2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(DropGNN_V2, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ELU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear4 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear3.reset_parameters()
        self.linear4.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()

    def forward(self, data):
        # edge_index = data.edge_index
        x = data.x
        batch = data.batch
        # print(f'x before view: {x.shape}')
        # x = x.view(-1, x.size(-1))
        # print(f'x after view: {x.shape}')
        x = self.bn1(self.linear1(x))
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # print(f'x after linear 1: {x.shape}')
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(self.linear2(x))
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # print(f'x after linear 2: {x.shape}')

        x = global_add_pool(x, batch)
        # print(f'x after sum pooling: {x.shape}')
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear3(x)
        x = self.activation(x)

        x = self.linear4(x)
        # print(f'x after linear 3: {x.shape}')
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
                                                        'PTC_GIN'], default='MUTAG',
                        help="Options are ['MUTAG', 'IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'ENZYMES', 'PTC_GIN']")
    # parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=1234, help='seed for reproducibility')
    # parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    # parser.add_argument('--dropout', type=float, choices=[0.5, 0.2], default=0.2, help='dropout probability')
    parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')
    # parser.add_argument('--min_delta', type=float, default=0.001, help='min_delta in early stopping')
    # parser.add_argument('--patience', type=int, default=100, help='patience in early stopping')
    parser.add_argument('--agg', type=str, default="mean", choices=["mean", "concat"],
                        help='Method for aggregating the perturbation')
    # parser.add_argument('--normalization', type=str, default='After', choices=['After', 'Before'],
    #                    help='Doing normalization before generation of perturbations or after')
    args = parser.parse_args()
    wandb.login()
    dataset = get_dataset(args)
    print(dataset)
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
    print(f'Max number of nodes: {torch.tensor(n).float().max().round().long().item()}')
    print(f'Min number of nodes: {torch.tensor(n).float().min().round().long().item()}')
    print(f'Number of graphs: {len(dataset)}')
    gamma = mean_n
    p = 2 * 1 / (1 + gamma)
    num_perturbations = round(gamma * np.log10(gamma))
    print(f'Number of perturbations: {num_perturbations}')
    print(f'Sampling probability: {p}')
    current_path = os.getcwd()

    with wandb.init(config=config):
        config = wandb.config
        start_time = time.time()
        enriched_dataset = EnrichedGraphDataset(os.path.join(current_path, 'enriched_dataset'), dataset, k=4, p=p,
                                                num_perturbations=num_perturbations, config=config, args=args)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {device}')
        seeds_to_test = [0, 64]
        n_splits = 10
        final_acc = []
        final_std = []

        for seed in seeds_to_test:
            print(f'Seed: {seed}')
            print("==============")
            torch.manual_seed(seed)
            np.random.seed(seed)
            all_validation_accuracies = []
            time_seed = []

            skf_splits = separate_data(len(dataset), seed)

            # Iterate through each fold
            for fold, (train_indices, test_indices) in enumerate(skf_splits):
                print(f'Fold {fold + 1}/{n_splits}:')
                start_time_fold = time.time()
                # Create data loaders for the current fold
                train_loader = DataLoader(
                    dataset[train_indices],
                    sampler=RandomSampler(dataset[train_indices], replacement=True,
                                          num_samples=int(
                                              len(train_indices) * 50 / (len(train_indices) / config.batch_size))),
                    batch_size=config.batch_size, drop_last=False,
                    collate_fn=Collater(follow_batch=[], exclude_keys=[]))

                test_loader = DataLoader(dataset[test_indices], batch_size=config.batch_size, shuffle=False)

                # Reinitialize the model for each fold
                model = DropGNN_V2(enriched_dataset.num_features, config.hidden_dim, enriched_dataset.num_classes,
                                   config.dropout).to(device)
                if fold == 0:
                    print(f'Model learnable parameters: {count_parameters(model)}')
                optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

                validation_accuracies = []
                # Training loop for the current fold
                for epoch in range(1, args.epochs + 1):
                    lr = scheduler.optimizer.param_groups[0]['lr']
                    train_loss = train(model, train_loader, optimizer, device)
                    scheduler.step()
                    test_acc = test(model, test_loader, device)
                    if epoch % 20 == 0:
                        print(f'Epoch: {epoch:02d} | TrainLoss: {train_loss:.3f} | Test_acc: {test_acc:.3f}')
                    validation_accuracies.append(test_acc)

                all_validation_accuracies.append(validation_accuracies)
                # Print fold training time
                end_time_fold = time.time()
                elapsed_time_fold = end_time_fold - start_time_fold
                print(f'Time taken for training in seed {seed}, fold {fold + 1}: {elapsed_time_fold:.2f} seconds')
                time_seed.append(elapsed_time_fold)
            print("======================================")
            average_validation_curve = np.mean(all_validation_accuracies, axis=0)
            max_avg_validation_acc_epoch = np.argmax(average_validation_curve)
            best_epoch_mean = average_validation_curve[max_avg_validation_acc_epoch]
            std_at_max_avg_validation_acc_epoch = np.std(
                [validation_accuracies[max_avg_validation_acc_epoch] for validation_accuracies in
                 all_validation_accuracies])

            final_acc.append(best_epoch_mean)
            final_std.append(std_at_max_avg_validation_acc_epoch)

            print(f'Epoch {max_avg_validation_acc_epoch + 1} got maximum averaged validation accuracy in seed {seed}: {best_epoch_mean}')
            print(f'Standard Deviation for the results of epoch {max_avg_validation_acc_epoch + 1} over all the folds '
                  f'in seed {seed}: {std_at_max_avg_validation_acc_epoch}')
            print(f'Average time taken for each fold in seed {seed}: {np.mean(time_seed)}')

        print("======================================")
        print(f'Test accuracy for all the seeds: {np.mean(final_acc)}')
        print(f'Std for all the seeds: {np.mean(final_std)}')
        wandb.log(
            {"Test Accuracy for all the seeds": np.mean(final_acc),
             "Std for all the seeds": np.mean(final_std)
             })


if __name__ == "__main__":
    wandb.agent(sweep_id, main, count=1)
