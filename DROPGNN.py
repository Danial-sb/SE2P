import time
import random
import logging
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler

from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, global_add_pool, GINConv

from test_tube import HyperOptArgumentParser

from SE2P import separate_data, train, test, count_parameters
from datasets import get_dataset


# Code is based on the Dropout Graph Neural Network (DropGNN) paper.

logging.basicConfig(filename='log/test.log', level=logging.INFO, format='%(asctime)s - %(message)s')


def log_and_print(message, log=False):
    print(message)
    if log:
        logging.info(message)


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, args: Any):
        super(GCN, self).__init__()

        hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.args = args

        self.num_layers = 4

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.fcs.append(nn.Linear(input_dim, output_dim))
        self.fcs.append(nn.Linear(hidden_dim, output_dim))

        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.fcs.append(nn.Linear(hidden_dim, output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, GCNConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data):
        x = data.x.to(torch.float)
        edge_index = data.edge_index
        batch = data.batch
        outs = [x]
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x)

        out = None
        for i, x in enumerate(outs):
            x = global_add_pool(x, batch)
            x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
            if out is None:
                out = x
            else:
                out += x

        if self.args.dataset in ['ogbg-moltox21', 'ogbg-molhiv']:
            return out
        else:
            return F.log_softmax(out, dim=-1)


class DropGCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_perturbations, p, args: Any):
        super(DropGCN, self).__init__()

        dim = args.hidden_dim
        self.args = args
        self.dropout = args.dropout
        self.num_perturbations = num_perturbations
        self.p = p

        self.num_layers = 4

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(GCNConv(input_dim, dim))
        self.bns.append(nn.BatchNorm1d(dim))
        self.fcs.append(nn.Linear(input_dim, output_dim))
        self.fcs.append(nn.Linear(dim, output_dim))

        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(dim, dim))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(dim, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, GCNConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data):
        x = data.x.to(torch.float)
        edge_index = data.edge_index
        batch = data.batch

        # Do runs in paralel, by repeating the graphs in the batch
        x = x.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()
        drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * self.p).bool()
        x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)
        del drop
        outs = [x]
        x = x.view(-1, x.size(-1))
        run_edge_index = edge_index.repeat(1, self.num_perturbations) + torch.arange(self.num_perturbations,
                                                                                     device=edge_index.device).repeat_interleave(
            edge_index.size(1)) * (edge_index.max() + 1)
        for i in range(self.num_layers):
            x = self.convs[i](x, run_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x.view(self.num_perturbations, -1, x.size(-1)))
        del run_edge_index
        out = None
        for i, x in enumerate(outs):
            x = x.mean(dim=0)
            x = global_add_pool(x, batch)
            x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
            if out is None:
                out = x
            else:
                out += x

        if self.args.dataset in ['ogbg-moltox21', 'ogbg-molhiv']:
            return out
        else:
            return F.log_softmax(out, dim=-1)


class GIN(nn.Module):
    def __init__(self, input_dim, output_dim, args: Any):
        super(GIN, self).__init__()

        dim = args.hidden_dim
        self.dropout = args.dropout
        self.args = args

        self.num_layers = 4

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(GINConv(
            nn.Sequential(nn.Linear(input_dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
        self.bns.append(nn.BatchNorm1d(dim))
        self.fcs.append(nn.Linear(input_dim, output_dim))
        self.fcs.append(nn.Linear(dim, output_dim))

        for i in range(self.num_layers - 1):
            self.convs.append(
                GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(dim, output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, GINConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data):
        x = data.x.to(torch.float)
        edge_index = data.edge_index
        batch = data.batch
        outs = [x]
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x)

        out = None
        for i, x in enumerate(outs):
            x = global_add_pool(x, batch)
            x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
            if out is None:
                out = x
            else:
                out += x

        if self.args.dataset in ['ogbg-moltox21', 'ogbg-molhiv']:
            return out
        else:
            return F.log_softmax(out, dim=-1)


class DropGIN(nn.Module):
    def __init__(self, input_dim, output_dim, num_perturbations, p, args: Any):
        super(DropGIN, self).__init__()

        dim = args.hidden_dim
        self.dropout = args.dropout
        self.num_perturbations = num_perturbations
        self.p = p
        self.args =args

        self.num_layers = 4

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(GINConv(
            nn.Sequential(nn.Linear(input_dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
        self.bns.append(nn.BatchNorm1d(dim))
        self.fcs.append(nn.Linear(input_dim, output_dim))
        self.fcs.append(nn.Linear(dim, output_dim))

        for i in range(self.num_layers - 1):
            self.convs.append(
                GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(dim, output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, GINConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, data):
        x = data.x.to(torch.float)
        edge_index = data.edge_index
        batch = data.batch

        # Do runs in paralel, by repeating the graphs in the batch
        x = x.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()
        drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * self.p).bool()
        x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)
        del drop
        outs = [x]
        x = x.view(-1, x.size(-1))
        run_edge_index = edge_index.repeat(1, self.num_perturbations) + torch.arange(self.num_perturbations,
                                                                                     device=edge_index.device).repeat_interleave(
            edge_index.size(1)) * (edge_index.max() + 1)
        for i in range(self.num_layers):
            x = self.convs[i](x, run_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x.view(self.num_perturbations, -1, x.size(-1)))
        del run_edge_index
        out = None
        for i, x in enumerate(outs):
            x = x.mean(dim=0)
            x = global_add_pool(x, batch)
            x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
            if out is None:
                out = x
            else:
                out += x

        if self.args.dataset in ['ogbg-moltox21', 'ogbg-molhiv']:
            return out
        else:
            return F.log_softmax(out, dim=-1)


def main(args):
    print(args, flush=True)
    dataset = get_dataset(args)

    n = []
    degs = []
    for g in dataset:
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
    num_runs = gamma
    print(f'Number of runs: {num_runs}')
    print(f'Sampling probability: {p}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    log_and_print(f'Device: {device}')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    n_splits = 10

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    log_and_print(f'Seed: {args.seed}')
    log_and_print('=' * 50)
    all_validation_accuracies = []
    time_seed = []

    skf_splits = separate_data(len(dataset), n_splits, args.seed)

    if args.model == "DropGIN":
        model = DropGIN(dataset.num_features, dataset.num_classes, num_runs, p, args).to(device)
    elif args.model == "GCN":
        model = GCN(dataset.num_features, dataset.num_classes, args).to(device)
    elif args.model == "DropGCN":
        model = DropGCN(dataset.num_features, dataset.num_classes, num_runs, p, args).to(device)
    else:
        model = GIN(dataset.num_features, dataset.num_classes, args).to(device)

    # Iterate through each folds
    for fold, (train_indices, test_indices) in enumerate(skf_splits):
        model.reset_parameters()
        log_and_print(f'Fold {fold + 1}/{n_splits}:')
        start_time_fold = time.time()
        # Create data loaders for the current fold
        train_loader = DataLoader(
            dataset[train_indices.tolist()],
            sampler=RandomSampler(dataset[train_indices.tolist()], replacement=True,
                                  num_samples=int(
                                      len(train_indices.tolist()) * 50 / (
                                              len(train_indices.tolist()) / args.batch_size)),
                                  generator=generator),
            batch_size=args.batch_size, drop_last=False,
            collate_fn=Collater(follow_batch=[], exclude_keys=[]))

        test_loader = DataLoader(dataset[test_indices.tolist()], batch_size=args.batch_size, shuffle=False)

        if fold == 0:
            log_and_print(f'Model learnable parameters for {model.__class__.__name__}: {count_parameters(model)}')

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        validation_accuracies = []
        time_per_epoch = []
        max_memory_allocated = 0
        max_memory_reserved = 0
        # Training loop for the current fold
        for epoch in range(1, args.epochs + 1):
            start_time_epoch = time.time()
            lr = scheduler.optimizer.param_groups[0]['lr']
            train_loss = train(model, train_loader, optimizer, device)
            scheduler.step()
            # torch.cuda.reset_peak_memory_stats()

            memory_allocated = torch.cuda.max_memory_allocated(device) // (1024 ** 2)
            memory_reserved = torch.cuda.max_memory_reserved(device) // (1024 ** 2)
            max_memory_allocated = max(max_memory_allocated, memory_allocated)
            max_memory_reserved = max(max_memory_reserved, memory_reserved)

            test_acc = test(model, test_loader, device)
            end_time_epoch = time.time()
            elapsed_time_epoch = end_time_epoch - start_time_epoch
            time_per_epoch.append(elapsed_time_epoch)
            if epoch % 1 == 0 or epoch == 1:
                log_and_print(f'Epoch: {epoch:02d} | TrainLoss: {train_loss:.3f} | Test_acc: {test_acc:.3f} | Time'
                              f'epoch: {elapsed_time_epoch:.2f} | Memory Allocated: {memory_allocated} MB | Memory '
                              f'Reserved: {memory_reserved} MB | LR: {lr:.6f}')
            validation_accuracies.append(test_acc)

        log_and_print(f'Average time per epoch in fold {fold + 1} and seed {args.seed}: {np.mean(time_per_epoch)}')
        log_and_print(f'Std time per epoch in fold {fold + 1} and seed {args.seed}: {np.std(time_per_epoch)}')
        all_validation_accuracies.append(torch.tensor(validation_accuracies))
        # Print fold training time
        end_time_fold = time.time()
        elapsed_time_fold = end_time_fold - start_time_fold
        log_and_print(f'Time taken for training in seed {args.seed}, fold {fold + 1}: {elapsed_time_fold:.2f} seconds, '
                      f'Max Memory Allocated: {max_memory_allocated} MB | Max Memory Reserved: {max_memory_reserved} MB')
        time_seed.append(elapsed_time_fold)
    log_and_print("=" * 50)
    average_validation_curve = torch.stack(all_validation_accuracies, dim=0)
    acc_mean = average_validation_curve.mean(dim=0)
    best_epoch = acc_mean.argmax().item()
    best_epoch_mean = average_validation_curve[:, best_epoch].mean()
    std_at_max_avg_validation_acc_epoch = average_validation_curve[:, best_epoch].std()

    log_and_print(f'Epoch {best_epoch + 1} got maximum averaged validation accuracy in seed {args.seed}:'
                  f'{best_epoch_mean}')
    log_and_print(f'Standard Deviation for the results of epoch {best_epoch + 1} over all the folds in '
                  f'seed {args.seed}: {std_at_max_avg_validation_acc_epoch}')
    log_and_print(f'Average time taken for each fold in seed {args.seed}: {np.mean(time_seed)}')
    log_and_print(f'STD time taken for each fold in seed {args.seed}: {np.std(time_seed)}')


if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.add_argument('--dataset', type=str, choices=['MUTAG', 'IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'ENZYMES',
                                                        'PTC_GIN', 'NCI109', 'COLLAB'], default='MUTAG',
                        help="Options are ['MUTAG', 'IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'ENZYMES', 'PTC_GIN']")
    parser.opt_list('--dropout', type=float, default=0.5, tunable=True, options=[0.5])
    parser.opt_list('--batch_size', type=int, default=12, tunable=True, options=[32])
    parser.opt_list('--hidden_dim', type=int, default=32, tunable=True, options=[16])
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--model', type=str, choices=['GIN', 'DropGIN', 'GCN', 'DropGCN'], default="DropGCN")
    parser.add_argument('--epochs', type=int, default=350, help='maximum number of epochs')
    parser.add_argument('--grid_search', action='store_true', default=False, help='whether to do grid search')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    log_and_print(f"model:{args.model}")

    if args.grid_search:
        log_and_print("Doing grid-search")
        for hparam_trial in args.trials(None):
            log_and_print(hparam_trial)
            main(hparam_trial)
    else:
        main(args)

    log_and_print('Finished')
