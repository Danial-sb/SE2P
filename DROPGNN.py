import numpy as np
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, global_add_pool, GINConv
from sklearn.model_selection import StratifiedKFold, KFold
import logging
import random
from test_tube import HyperOptArgumentParser
from datasets import get_dataset

logging.basicConfig(filename='log/DropGCN_IMDBM_NEW.log', level=logging.INFO, format='%(asctime)s - %(message)s')


def main(args, cluster=None):
    print(args, flush=True)

    BATCH = args.batch_size
    dataset = get_dataset(args)

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

    def separate_data(dataset_len, n_splits, seed=0):
        # Use same splitting/10-fold as GIN paper
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        idx_list = []
        for idx in skf.split(np.zeros(dataset_len), np.zeros(dataset_len)):
            idx_list.append(idx)
        return idx_list

    # class CustomGCNConv(nn.Module):
    #     def __init__(self, input_dim, output_dim):
    #         super(CustomGCNConv, self).__init__()
    #         self.linear_1 = nn.Linear(input_dim, output_dim)
    #         self.bn = nn.BatchNorm1d(output_dim)
    #         self.relu = nn.ReLU()
    #         self.linear_2 = nn.Linear(output_dim, output_dim)
    #         self.gcnconv = GCNConv(output_dim, output_dim)  # Using GCNConv with specified dimensions
    #
    #     def forward(self, x, edge_index):
    #         x = self.linear_1(x)
    #         x = self.bn(x)
    #         x = self.relu(x)
    #         x = self.linear_2(x)
    #         x = self.gcnconv(x, edge_index)
    #         return x

    class GCN(nn.Module):
        def __init__(self):
            super(GCN, self).__init__()

            num_features = dataset.num_features
            hidden_dim = args.hidden_units
            self.dropout = args.dropout

            self.num_layers = 4

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(GCNConv(num_features, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.fcs.append(nn.Linear(num_features, dataset.num_classes))
            self.fcs.append(nn.Linear(hidden_dim, dataset.num_classes))

            for i in range(self.num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
                self.fcs.append(nn.Linear(hidden_dim, dataset.num_classes))

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

    # class GCN(nn.Module):
    #     def __init__(self):
    #         super(GCN, self).__init__()
    #
    #         num_features = dataset.num_features
    #         dim = args.hidden_units
    #         self.dropout = args.dropout
    #
    #         self.num_layers = 4
    #
    #         self.convs = nn.ModuleList()
    #         self.bns = nn.ModuleList()
    #         self.fcs = nn.ModuleList()
    #
    #         self.convs.append(CustomGCNConv(num_features, dim))
    #         self.bns.append(nn.BatchNorm1d(dim))
    #         self.fcs.append(nn.Linear(num_features, dataset.num_classes))
    #         self.fcs.append(nn.Linear(dim, dataset.num_classes))
    #
    #         for i in range(self.num_layers - 1):
    #             self.convs.append(CustomGCNConv(dim, dim))
    #             self.bns.append(nn.BatchNorm1d(dim))
    #             self.fcs.append(nn.Linear(dim, dataset.num_classes))
    #         self.reset_parameters()
    #
    #     def reset_parameters(self):
    #         for m in self.modules():
    #             if isinstance(m, nn.Linear) or isinstance(m, GCNConv):
    #                 m.reset_parameters()
    #             elif isinstance(m, nn.BatchNorm1d):
    #                 m.reset_parameters()
    #
    #     def forward(self, data):
    #         x = data.x
    #         edge_index = data.edge_index
    #         batch = data.batch
    #         outs = [x]
    #         for i in range(self.num_layers):
    #             x = self.convs[i](x, edge_index)
    #             x = self.bns[i](x)
    #             x = F.relu(x)
    #             outs.append(x)
    #
    #         out = None
    #         for i, x in enumerate(outs):
    #             x = global_add_pool(x, batch)
    #             x = F.dropout(self.fcs[i](x), p=self.dropout, training=self.training)
    #             if out is None:
    #                 out = x
    #             else:
    #                 out += x
    #         return F.log_softmax(out, dim=-1)

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

            self.convs.append(GCNConv(num_features, dim))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, dataset.num_classes))
            self.fcs.append(nn.Linear(dim, dataset.num_classes))

            for i in range(self.num_layers - 1):
                self.convs.append(GCNConv(dim, dim))
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, dataset.num_classes))
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

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f'Device: {device}')
    logging.info(f'Device: {device}')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    n_splits = 10
    # final_acc = []
    # final_std = []

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    print(f'Seed: {args.seed}')
    print("==============")
    logging.info(f'Seed: {args.seed}')
    logging.info("==============")
    all_validation_accuracies = []
    time_seed = []
    # acc = []
    # splits = separate_data(len(dataset), n_splits, seed=args.seed)
    # model = DropGIN().to(device)
    # print(model.__class__.__name__)
    # for i, (train_idx, test_idx) in enumerate(splits):
    #     model.reset_parameters()
    #     lr = args.lr
    #     BATCH = args.batch_size
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50,
    #                                                 gamma=0.5)  # in GIN code 50 itters per epoch were used
    #
    #     test_dataset = dataset[test_idx.tolist()]
    #     train_dataset = dataset[train_idx.tolist()]
    #
    #     test_loader = DataLoader(test_dataset, batch_size=BATCH)
    #     train_loader = torch.utils.data.DataLoader(train_dataset,
    #                                                sampler=torch.utils.data.RandomSampler(train_dataset,
    #                                                                                       replacement=True,
    #                                                                                       num_samples=int(
    #                                                                                           len(train_dataset) * 50 / (
    #                                                                                                       len(train_dataset) / BATCH)),
    #                                                                                       generator=generator),
    #                                                batch_size=BATCH, drop_last=False,
    #                                                collate_fn=Collater(follow_batch=[],
    #                                                                    exclude_keys=[]))  # GIN like epochs/batches - they do 50 radom batches per epoch
    #
    #     print('---------------- Split {} ----------------'.format(i), flush=True)
    #
    #     test_acc = 0
    #     acc_temp = []
    #     for epoch in range(1, args.epochs + 1):
    #         if epoch == args.epochs:
    #             start = time.time()
    #         lr = scheduler.optimizer.param_groups[0]['lr']
    #         train_loss = train(model, train_loader, optimizer, device)
    #         scheduler.step()
    #         test_acc = test(model, test_loader, device)
    #         if epoch == args.epochs:
    #             print('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
    #                   'Val Loss: {:.7f}, Test Acc: {:.7f}, Time: {:7f}'.format(
    #                 epoch, lr, train_loss, 0, test_acc, time.time() - start), flush=True)
    #             logging.info('Epoch: {:03d}, LR: {:7f}, Train Loss: {:.7f}, '
    #                   'Val Loss: {:.7f}, Test Acc: {:.7f}, Time: {:7f}'.format(
    #                 epoch, lr, train_loss, 0, test_acc, time.time() - start))
    #         acc_temp.append(test_acc)
    #     acc.append(torch.tensor(acc_temp))
    # acc = torch.stack(acc, dim=0)
    # acc_mean = acc.mean(dim=0)
    # best_epoch = acc_mean.argmax().item()
    # print('---------------- Final Epoch Result ----------------')
    # logging.info('---------------- Final Epoch Result ----------------')
    # print('Mean: {:7f}, Std: {:7f}'.format(acc[:, -1].mean(), acc[:, -1].std()))
    # logging.info('Mean: {:7f}, Std: {:7f}'.format(acc[:, -1].mean(), acc[:, -1].std()))
    # print(f'---------------- Best Epoch: {best_epoch} ----------------')
    # logging.info(f'---------------- Best Epoch: {best_epoch} ----------------')
    # print('Mean: {:7f}, Std: {:7f}'.format(acc[:, best_epoch].mean(), acc[:, best_epoch].std()))
    # logging.info('Mean: {:7f}, Std: {:7f}'.format(acc[:, best_epoch].mean(), acc[:, best_epoch].std()))

    skf_splits = separate_data(len(dataset), n_splits, args.seed)

    if args.model == "DropGIN":
        model = DropGIN().to(device)
    elif args.model == "GCN":
        model = GCN().to(device)
    elif args.model == "DropGCN":
        model = DropGCN().to(device)
    else:
        model = GIN().to(device)

    # Iterate through each folds
    for fold, (train_indices, test_indices) in enumerate(skf_splits):
        model.reset_parameters()
        print(f'Fold {fold + 1}/{n_splits}:')
        logging.info(f'Fold {fold + 1}/{n_splits}:')
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
            print(f'Model Parameters: {count_parameters(model)}')
            logging.info(f'Model Parameters: {count_parameters(model)}')

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
            if epoch % 50 == 0 or epoch == 1:
                print(f'Epoch: {epoch:02d} | TrainLoss: {train_loss:.3f} | Test_acc: {test_acc:.3f} | Time'
                      f'epoch: {elapsed_time_epoch:.2f} | Memory Allocated: {memory_allocated} MB | Memory '
                      f'Reserved: {memory_reserved} MB | LR: {lr:.6f}')
                logging.info(f'Epoch: {epoch:02d} | TrainLoss: {train_loss:.3f} | Test_acc: {test_acc:.3f} | Time'
                             f'epoch: {elapsed_time_epoch:.2f} | Memory Allocated: {memory_allocated} MB | Memory '
                             f'Reserved: {memory_reserved} MB | LR: {lr:.6f}')
            validation_accuracies.append(test_acc)

        print(f'Average time per epoch in fold {fold + 1} and seed {args.seed}: {np.mean(time_per_epoch)}')
        print(f'Std time per epoch in fold {fold + 1} and seed {args.seed}: {np.std(time_per_epoch)}')
        logging.info(f'Average time per epoch in fold {fold + 1} and seed {args.seed}: {np.mean(time_per_epoch)}')
        logging.info(f'Std time per epoch in fold {fold + 1} and seed {args.seed}: {np.std(time_per_epoch)}')
        all_validation_accuracies.append(torch.tensor(validation_accuracies))
        # Print fold training time
        end_time_fold = time.time()
        elapsed_time_fold = end_time_fold - start_time_fold
        print(f'Time taken for training in seed {args.seed}, fold {fold + 1}: {elapsed_time_fold:.2f} seconds, '
              f'Max Memory Allocated: {max_memory_allocated} MB | Max Memory Reserved: {max_memory_reserved} MB')
        logging.info(f'Time taken for training in seed {args.seed}, fold {fold + 1}: {elapsed_time_fold:.2f} seconds, '
                     f'Max Memory Allocated: {max_memory_allocated} MB | Max Memory Reserved: {max_memory_reserved} MB')
        time_seed.append(elapsed_time_fold)
    print("======================================")
    logging.info("======================================")
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

    print(
        f'Epoch {best_epoch + 1} got maximum averaged validation accuracy in seed {args.seed}: {best_epoch_mean}')
    print(f'Standard Deviation for the results of epoch {best_epoch + 1} over all the folds in '
          f'seed {args.seed}: {std_at_max_avg_validation_acc_epoch}')
    print(f'Average time taken for each fold in seed {args.seed}: {np.mean(time_seed)}')
    print(f'STD time taken for each fold in seed {args.seed}: {np.std(time_seed)}')
    logging.info(
        f'Epoch {best_epoch + 1} got maximum averaged validation accuracy in seed {args.seed}: {best_epoch_mean}')
    logging.info(
        f'Standard Deviation for the results of epoch {best_epoch + 1} over all the folds in '
        f'seed {args.seed}: {std_at_max_avg_validation_acc_epoch}')
    logging.info(f'Average time taken for each fold in seed {args.seed}: {np.mean(time_seed)}')
    logging.info(f'STD time taken for each fold in seed {args.seed}: {np.std(time_seed)}')


#
# # print("======================================")
# print(f'Test accuracy for all the seeds: {np.mean(final_acc)}')
# print(f'Std for all the seeds: {np.mean(final_std)}')
# logging.info("======================================")
# logging.info(f'Test accuracy for all the seeds: {np.mean(final_acc)}')
# logging.info(f'Std for all the seeds: {np.mean(final_std)}')


if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.add_argument('--dataset', type=str, choices=['MUTAG', 'IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'ENZYMES',
                                                        'PTC_GIN', 'NCI109', 'COLLAB'], default='IMDB-MULTI',
                        help="Options are ['MUTAG', 'IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'ENZYMES', 'PTC_GIN']")
    parser.opt_list('--dropout', type=float, default=0.5, tunable=True, options=[0.5, 0.0])
    parser.opt_list('--batch_size', type=int, default=32, tunable=True, options=[32, 64])
    parser.opt_list('--hidden_units', type=int, default=32, tunable=True, options=[32, 64])
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    # parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    # parser.add_argument('--seed', type=int, default=1234, help='seed for reproducibility')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--model', type=str, choices=['GIN', 'DropGIN', 'GCN', 'DropGCN'], default="DropGCN")
    # parser.add_argument('--hidden_units', type=int, default=64, choices=[32, 64])
    # parser.add_argument('--dropout', type=float, choices=[0.5, 0.2], default=0.5, help='dropout probability')
    parser.add_argument('--epochs', type=int, default=350, help='maximum number of epochs')
    parser.add_argument('--grid_search', action='store_true', default=True, help='whether to do grid search')
    # parser.add_argument('--min_delta', type=float, default=0.001, help='min_delta in early stopping')
    # parser.add_argument('--patience', type=int, default=100, help='patience in early stopping')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"model:{args.model}")
    logging.info(f"model:{args.model}")

    if args.grid_search:
        print("Doing grid-search")
        logging.info("Doing grid-search")
        for hparam_trial in args.trials(None):
            print(hparam_trial)
            logging.info(hparam_trial)
            main(hparam_trial)
    else:
        main(args)

    print('Finished', flush=True)
    logging.info("Finished")
