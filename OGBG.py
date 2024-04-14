from SDGNN import *
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ogb.graphproppred import Evaluator
from torch_geometric.utils import degree
import time
from torch_geometric.nn import GCNConv, global_add_pool, GINConv
import argparse
import wandb
import os.path as osp


def get_ogb(args):
    if 'ogbg-molhiv' in args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ogbg-molhiv')
        dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=path)
    elif 'ogbg-molpcba' in args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ogbg-molpcba')
        dataset = PygGraphPropPredDataset(name="ogbg-molpcba", root=path)
    elif 'ogbg-moltox21' in args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ogbg-moltox21')
        dataset = PygGraphPropPredDataset(name="ogbg-moltox21", root=path)
    else:
        raise ValueError("Invalid dataset name")

    return dataset


class SDGNN_ogb(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, num_layers, batch_norm=True):
        super(SDGNN_ogb, self).__init__()

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

        self.activation = nn.ELU()
        self.linear3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.linear4 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()
        self.linear3.reset_parameters()
        self.linear4.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, data):
        x = data.x
        batch = data.batch

        for i in range(self.num_layers):
            x = self.linears[i](x)
            if self.batch_norm:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_add_pool(x, batch)

        x = self.linear3(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear4(x)
        return x


class GIN_ogb(nn.Module):
    def __init__(self, config, dataset, output):
        super(GIN_ogb, self).__init__()

        num_features = dataset.num_features
        dim = config.hidden_dim
        self.dropout = config.dropout

        self.num_layers = 4

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(GINConv(
            nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
        self.bns.append(nn.BatchNorm1d(dim))
        self.fcs.append(nn.Linear(num_features, output))
        self.fcs.append(nn.Linear(dim, output))

        for i in range(self.num_layers - 1):
            self.convs.append(
                GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(dim, output))

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
        return out


class CustomGCNConv_ogb(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomGCNConv_ogb, self).__init__()
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


class GCN_ogb(nn.Module):
    def __init__(self, config, dataset, output):
        super(GCN_ogb, self).__init__()

        num_features = dataset.num_features
        dim = config.hidden_dim
        self.dropout = config.dropout

        self.num_layers = 4

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(CustomGCNConv_ogb(num_features, dim))
        self.bns.append(nn.BatchNorm1d(dim))
        self.fcs.append(nn.Linear(num_features, output))
        self.fcs.append(nn.Linear(dim, output))

        for i in range(self.num_layers - 1):
            self.convs.append(CustomGCNConv_ogb(dim, dim))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(dim, output))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, GCNConv):
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
        return out


class DropGCN_ogb(nn.Module):
    def __init__(self, config, dataset, num_runs, p, output):
        super(DropGCN_ogb, self).__init__()

        self.num_runs = num_runs
        self.p = p
        num_features = dataset.num_features
        dim = config.hidden_dim
        self.dropout = config.dropout

        self.num_layers = 4

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(CustomGCNConv_ogb(num_features, dim))
        self.bns.append(nn.BatchNorm1d(dim))
        self.fcs.append(nn.Linear(num_features, output))
        self.fcs.append(nn.Linear(dim, output))

        for i in range(self.num_layers - 1):
            self.convs.append(CustomGCNConv_ogb(dim, dim))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(dim, output))
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
        x = x.unsqueeze(0).expand(self.num_runs, -1, -1).clone()
        drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * self.p).bool()
        x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)
        del drop
        outs = [x]
        x = x.view(-1, x.size(-1))
        run_edge_index = edge_index.repeat(1, self.num_runs) + torch.arange(self.num_runs,
                                                                            device=edge_index.device).repeat_interleave(
            edge_index.size(1)) * (edge_index.max() + 1)
        for i in range(self.num_layers):
            x = self.convs[i](x, run_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x.view(self.num_runs, -1, x.size(-1)))
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

        return out


class DropGIN_ogb(nn.Module):
    def __init__(self, config, dataset, num_runs, p, output):
        super(DropGIN_ogb, self).__init__()

        self.num_runs = num_runs
        self.p = p
        num_features = dataset.num_features
        dim = config.hidden_dim
        self.dropout = config.dropout

        self.num_layers = 4

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.fcs = nn.ModuleList()

        self.convs.append(GINConv(
            nn.Sequential(nn.Linear(num_features, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
        self.bns.append(nn.BatchNorm1d(dim))
        self.fcs.append(nn.Linear(num_features, output))
        self.fcs.append(nn.Linear(dim, output))

        for i in range(self.num_layers - 1):
            self.convs.append(
                GINConv(nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim))))
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(dim, output))

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

        # Do runs in parallel, by repeating the graphs in the batch
        x = x.unsqueeze(0).expand(self.num_runs, -1, -1).clone()
        drop = torch.bernoulli(torch.ones([x.size(0), x.size(1)], device=x.device) * self.p).bool()
        x[drop] = torch.zeros([drop.sum().long().item(), x.size(-1)], device=x.device)
        del drop
        outs = [x]
        x = x.view(-1, x.size(-1))
        run_edge_index = edge_index.repeat(1, self.num_runs) + torch.arange(self.num_runs,
                                                                            device=edge_index.device).repeat_interleave(
            edge_index.size(1)) * (edge_index.max() + 1)
        for i in range(self.num_layers):
            x = self.convs[i](x, run_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            outs.append(x.view(self.num_runs, -1, x.size(-1)))
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
        return out


class DeepSet_ogb(nn.Module):
    def __init__(self, input_size, hidden_size, num_perturbations, max_nodes):
        super(DeepSet_ogb, self).__init__()

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
        x = self.mlp_perturbation(input_data)

        x_transformed = x.view(-1, self.num_perturbations, self.max_nodes, self.hidden_size)
        aggregated_output = torch.sum(x_transformed, dim=1)
        aggregated_output = aggregated_output.view(-1, self.hidden_size)

        final_output = self.mlp_aggregation(aggregated_output)

        return final_output


class SDGNN_Deepset_ogb(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_perturbations, max_nodes, output_dim):
        super(SDGNN_Deepset_ogb, self).__init__()

        self.num_perturbations = num_perturbations

        self.deepset_aggregator = DeepSet_ogb(input_dim, hidden_dim, num_perturbations, max_nodes)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.elu = nn.ELU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.linear3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.linear4 = nn.Linear(hidden_dim // 4, output_dim)
        self.dropout = dropout

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

        aggregated_features = self.deepset_aggregator(x)
        batch = batch.view(-1, self.num_perturbations).to(torch.float).mean(dim=1).long()

        x = self.elu(self.bn1(self.linear1(aggregated_features)))

        x = self.elu(self.bn2(self.linear2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_add_pool(x, batch)

        x = self.elu(self.bn3(self.linear3(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear4(x)

        return x


def train_ogb(train_loader, model, optimizer, device):
    total_loss = 0
    N = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        mask = ~torch.isnan(data.y)
        out = model(data)[mask]
        y = data.y[mask].to(torch.float)
        loss = criterion(out, y)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        N += data.num_graphs
        optimizer.step()
    return total_loss / N


def test_ogb(loader, model, evaluator, device):
    y_preds, y_trues = [], []
    for data in loader:
        data = data.to(device)
        y_preds.append(model(data))
        y_trues.append(data.y)

    return evaluator.eval({
        'y_pred': torch.cat(y_preds, dim=0),
        'y_true': torch.cat(y_trues, dim=0),
    })[evaluator.eval_metric]


def main(config=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['ogbg-molhiv', 'ogbg-molpcba', "ogbg-moltox21"], default='ogbg-moltox21')
    parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
    parser.add_argument('--model', type=str, choices=['SDGNN_ogb', 'GIN_ogb', 'GCN_ogb', 'DropGIN_ogb',
                                                      'DropGCN_ogb', 'SDGNN_deepset_ogb'], default='DropGCN_ogb')
    # parser.add_argument('--dropout', type=float, choices=[0.5, 0.0], default=0.5, help='dropout probability')
    parser.add_argument('--seed', type=int, default=0, help='seed for reproducibility')
    parser.add_argument('--agg', type=str, default="mean", choices=["mean", "concat", "deepset"],
                        help='Method for aggregating the perturbation')
    args = parser.parse_args()

    dataset = get_ogb(args)
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
    seeds_to_test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    all_validation_acc = []
    all_test_acc = []

    with wandb.init(config=config):
        for seed in seeds_to_test:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            config = wandb.config
            print(args)

            if args.model == 'SDGNN_ogb' or args.model == 'SDGNN_deepset_ogb':
                name = f"enriched_{args.dataset}_{args.agg}"
                start_time = time.time()
                enriched_dataset = EnrichedGraphDataset(os.path.join(current_path, 'enriched_dataset'), name, dataset, p=p,
                                                        num_perturbations=num_perturbations, max_nodes=max_nodes,
                                                        config=config,
                                                        args=args)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Done! Time taken: {elapsed_time:.2f} seconds")

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # device = torch.device('cpu')
            print(f'Device: {device}')
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            split_idx = dataset.get_idx_split()
            if args.model == 'SDGNN_ogb' or args.model == 'SDGNN_deepset_ogb':
                train_loader = DataLoader(enriched_dataset[split_idx["train"]], batch_size=config.batch_size, shuffle=True)
                valid_loader = DataLoader(enriched_dataset[split_idx["valid"]], batch_size=config.batch_size, shuffle=False)
                test_loader = DataLoader(enriched_dataset[split_idx["test"]], batch_size=config.batch_size, shuffle=False)
            else:
                train_loader = DataLoader(dataset[split_idx["train"]], batch_size=config.batch_size, shuffle=True)
                valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=config.batch_size, shuffle=False)
                test_loader = DataLoader(dataset[split_idx["test"]], batch_size=config.batch_size, shuffle=False)

            if args.model == 'SDGNN_ogb':
                model = SDGNN_ogb(enriched_dataset.num_features, config.hidden_dim,
                                  1 if args.dataset == 'ogbg-molhiv' else (12 if args.dataset == 'ogbg-moltox21' else 128),
                                  config.dropout, config.num_layers, config.batch_norm).to(device)
            elif args.model == 'GIN_ogb':
                model = GIN_ogb(config, dataset, output=1 if args.dataset == 'ogbg-molhiv' else (12 if args.dataset == 'ogbg-moltox21' else 128)).to(device)
            elif args.model == 'GCN_ogb':
                model = GCN_ogb(config, dataset, output=1 if args.dataset == 'ogbg-molhiv' else (12 if args.dataset == 'ogbg-moltox21' else 128)).to(device)
            elif args.model == 'DropGIN_ogb':
                model = DropGIN_ogb(config, dataset, num_perturbations, p,
                                    output=1 if args.dataset == 'ogbg-molhiv' else (12 if args.dataset == 'ogbg-moltox21' else 128)).to(
                    device)
            elif args.model == 'DropGCN_ogb':
                model = DropGCN_ogb(config, dataset, num_perturbations, p,
                                    output=1 if args.dataset == 'ogbg-molhiv' else (12 if args.dataset == 'ogbg-moltox21' else 128)).to(
                    device)
            else:
                model = SDGNN_Deepset_ogb(enriched_dataset.num_features, config.hidden_dim, config.dropout,
                                          num_perturbations, max_nodes,
                                          output_dim=1 if args.dataset == 'ogbg-molhiv' else (12 if args.dataset == 'ogbg-moltox21' else 128)).to(device)

            evaluator = Evaluator(name=args.dataset)

            print(f'Model learnable parameters: {count_parameters(model)}')
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

            start_outer = time.time()
            best_val_perf = test_perf = float('-inf')
            # counter = 0
            # patience = 80
            max_memory_allocated = 0
            max_memory_reserved = 0
            validation_perf = []
            tests_perf = []
            print(f'seed: {seed}')
            print("=" * 10)

            for epoch in range(1, args.epochs + 1):
                start = time.time()
                model.train()
                train_loss = train_ogb(train_loader, model, optimizer, device=device)
                wandb.log({"train_loss": train_loss})
                scheduler.step()

                memory_allocated = torch.cuda.max_memory_allocated(device) // (1024 ** 2)
                memory_reserved = torch.cuda.max_memory_reserved(device) // (1024 ** 2)
                max_memory_allocated = max(max_memory_allocated, memory_allocated)
                max_memory_reserved = max(max_memory_reserved, memory_reserved)

                model.eval()
                val_perf = test_ogb(valid_loader, model, evaluator, device)
                wandb.log({"val_perf": val_perf})
                validation_perf.append(val_perf)
                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    test_perf = test_ogb(test_loader, model, evaluator, device)
                tests_perf.append(test_perf)
                #     counter = 0
                # else:
                #     counter += 1
                #     if counter >= patience:  # maybe remove when having several seeds
                #         print(f'Early stopping at epoch {epoch} as no improvement seen in {patience} epochs.')
                #         break

                time_per_epoch = time.time() - start

                if epoch % 25 == 0 or epoch == 1:
                    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                          f'Val: {val_perf:.4f}, Test: {test_perf:.4f}, Seconds: {time_per_epoch:.4f},'
                          f' Memory allocated: {memory_allocated}, Memory Reserved: {memory_reserved}')

            time_average_epoch = time.time() - start_outer
            print(f'Best Validation in seed {seed}: {best_val_perf}, Test in seed {seed}: {test_perf}, Seconds/epoch: {time_average_epoch / args.epochs},'
                  f' Max memory allocated: {max_memory_allocated}, Max memory reserved: {max_memory_reserved}')
            # wandb.log(
            #     {f'Best Validation in seed {seed}': best_val_perf,
            #      f'Best Test in seed {seed}': test_perf}
            # )
            print("=" * 50)

            all_validation_acc.append(torch.tensor(validation_perf))
            all_test_acc.append(torch.tensor(tests_perf))
        final_vals = torch.stack(all_validation_acc)
        final_tests = torch.stack(all_test_acc)
        val_mean = final_vals.mean(dim=0)
        best_epoch = val_mean.argmax().item()
        best_epoch_mean_val = final_vals[:, best_epoch].mean()
        best_epoch_std_val = final_vals[:, best_epoch].std()
        best_epoch_mean_test = final_tests[:, best_epoch].mean()
        best_epoch_std_test = final_tests[:, best_epoch].std()

        print(f'Epoch {best_epoch + 1} got maximum average validation accuracy')
        print(f'Validation accuracy for all the seeds: {best_epoch_mean_val} | Std validation for all the seeds: '
              f'{best_epoch_std_val} | Test accuracy for all the seeds: {best_epoch_mean_test} | Std test for all the'
              f' seeds: {best_epoch_std_test}')

        wandb.log(
            {'Best Validation in all seeds': best_epoch_mean_val,
             'Std validation': best_epoch_std_val,
             'Best Test in all seeds': best_epoch_mean_test,
             'Std test': best_epoch_std_test}
        )


if __name__ == "__main__":
    wandb.agent(sweep_id, main)
