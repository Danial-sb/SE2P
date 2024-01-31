import argparse
import os.path as osp
import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, global_add_pool, GINConv
from sklearn.model_selection import StratifiedKFold, KFold


def main(args, cluster=None):
    print(args, flush=True)

    BATCH = args.batch_size

    if 'IMDB' in args.dataset:  # IMDB-BINARY or #IMDB-MULTI
        class MyFilter(object):
            def __call__(self, data):
                return data.num_nodes <= 70

        class MyPreTransform(object):
            def __call__(self, data):
                data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
                data.x = F.one_hot(data.x, num_classes=69).to(torch.float)  # 136 in k-gnn?
                return data

        path = osp.join(
            osp.dirname(osp.realpath(__file__)), 'data', f'{args.dataset}')
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

    else:
        raise ValueError

    print(dataset)

    # Set the sampling probability and number of runs/samples for the DropGIN
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
    num_runs = gamma
    print(f'Number of runs: {num_runs}')
    print(f'Sampling probability: {p}')

    def separate_data(dataset_len, seed=0):
        # Use same splitting/10-fold as GIN paper
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        idx_list = []
        for idx in skf.split(np.zeros(dataset_len), np.zeros(dataset_len)):
            idx_list.append(idx)
        return idx_list

    class CustomGCNConv(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(CustomGCNConv, self).__init__()
            self.linear_1 = nn.Linear(input_dim, output_dim)
            self.bn = nn.BatchNorm1d(output_dim)
            self.relu = nn.ReLU()
            self.linear_2 = nn.Linear(output_dim, output_dim)
            self.gcnconv = GCNConv(output_dim, output_dim)  # Using GCNConv with specified dimensions

        def forward(self, x, edge_index):
            x = self.linear_1(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.linear_2(x)
            x = self.gcnconv(x, edge_index)
            return x

    class GCN(nn.Module):
        def __init__(self):
            super(GCN, self).__init__()

            num_features = dataset.num_features
            dim = args.hidden_units
            self.dropout = args.dropout

            self.num_layers = 4

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(CustomGCNConv(num_features, dim))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, dataset.num_classes))
            self.fcs.append(nn.Linear(dim, dataset.num_classes))

            for i in range(self.num_layers - 1):
                self.convs.append(CustomGCNConv(dim, dim))
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, dataset.num_classes))

        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear) or isinstance(m, GCNConv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
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
            return F.log_softmax(out, dim=-1)

    class DropGCN(nn.Module):
        def __init__(self):
            super(DropGCN, self).__init__()

            num_features = dataset.num_features
            dim = args.hidden_units
            self.dropout = args.dropout

            self.num_layers = 4

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(CustomGCNConv(num_features, dim))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, dataset.num_classes))
            self.fcs.append(nn.Linear(dim, dataset.num_classes))

            for i in range(self.num_layers - 1):
                self.convs.append(CustomGCNConv(dim, dim))
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, dataset.num_classes))

        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, GCNConv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch

            # Do runs in paralel, by repeating the graphs in the batch
            x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
            drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * p).bool()
            x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)
            del drop
            outs = [x]
            x = x.view(-1, x.size(-1))
            run_edge_index = edge_index.repeat(1, num_runs) + torch.arange(num_runs,
                                                                           device=edge_index.device).repeat_interleave(
                edge_index.size(1)) * (edge_index.max() + 1)
            for i in range(self.num_layers):
                x = self.convs[i](x, run_edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x.view(num_runs, -1, x.size(-1)))
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

            return F.log_softmax(out, dim=-1)

    class GIN(nn.Module):
        def __init__(self):
            super(GIN, self).__init__()

            num_features = dataset.num_features
            dim = args.hidden_units
            self.dropout = args.dropout

            self.num_layers = 4

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(GINConv(
                nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, dataset.num_classes))
            self.fcs.append(nn.Linear(dim, dataset.num_classes))

            for i in range(self.num_layers - 1):
                self.convs.append(
                    GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, dataset.num_classes))

        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, GINConv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
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
            return F.log_softmax(out, dim=-1)

    class DropGIN(nn.Module):
        def __init__(self):
            super(DropGIN, self).__init__()

            num_features = dataset.num_features
            dim = args.hidden_units
            self.dropout = args.dropout

            self.num_layers = 4

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(GINConv(
                nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, dataset.num_classes))
            self.fcs.append(nn.Linear(dim, dataset.num_classes))

            for i in range(self.num_layers - 1):
                self.convs.append(
                    GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, dataset.num_classes))

        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, GINConv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch

            # Do runs in paralel, by repeating the graphs in the batch
            x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
            drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * p).bool()
            x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)
            del drop
            outs = [x]
            x = x.view(-1, x.size(-1))
            run_edge_index = edge_index.repeat(1, num_runs) + torch.arange(num_runs,
                                                                           device=edge_index.device).repeat_interleave(
                edge_index.size(1)) * (edge_index.max() + 1)
            for i in range(self.num_layers):
                x = self.convs[i](x, run_edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x.view(num_runs, -1, x.size(-1)))
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
            return F.log_softmax(out, dim=-1)

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    seeds_to_test = [0, 64]
    n_splits = 10
    final_acc = []
    final_std = []

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
                                          len(train_indices) * 50 / (len(train_indices) / args.batch_size))),
                batch_size=args.batch_size, drop_last=False,
                collate_fn=Collater(follow_batch=[], exclude_keys=[]))

            test_loader = DataLoader(dataset[test_indices], batch_size=args.batch_size, shuffle=False)

            # Reinitialize the model for each fold
            if args.model == "DropGIN":
                model = DropGIN().to(device)
            elif args.model == "GCN":
                model = GCN().to(device)
            elif args.model == "DropGCN":
                model = DropGCN().to(device)
            else:
                model = GIN().to(device)

            if fold == 0:
                print(f'Model Parameters: {count_parameters(model)}')

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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

        print(
            f'Epoch {max_avg_validation_acc_epoch + 1} got maximum averaged validation accuracy in seed {seed}: {best_epoch_mean}')
        print(f'Standard Deviation for the results of epoch {max_avg_validation_acc_epoch + 1} over all the folds in '
              f'seed {seed}: {std_at_max_avg_validation_acc_epoch}')
        print(f'Average time taken for each fold in seed {seed}: {np.mean(time_seed)}')

    print("======================================")
    print(f'Test accuracy for all the seeds: {np.mean(final_acc)}')
    print(f'Std for all the seeds: {np.mean(final_std)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['MUTAG', 'IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'ENZYMES'],
                        default='MUTAG', help="Options are ['MUTAG', 'IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS']")
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--seed', type=int, default=1234, help='seed for reproducibility')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--model', type=str, choices=['GIN', 'DropGIN', 'GCN', 'DropGCN'], default="GIN")
    parser.add_argument('--hidden_units', type=int, default=32, choices=[64, 32])
    parser.add_argument('--dropout', type=float, choices=[0.5, 0.2], default=0.2, help='dropout probability')
    parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')
    parser.add_argument('--min_delta', type=float, default=0.001, help='min_delta in early stopping')
    parser.add_argument('--patience', type=int, default=100, help='patience in early stopping')

    args = parser.parse_args()
    print(f"model:{args.model}")
    main(args)

    print('Finished', flush=True)
