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
    else:
        raise ValueError("Invalid dataset name")

    return dataset

def compute_symmetric_normalized_adj(edge_index):
  # Convert to dense adjacency matrix
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
    all_adj = []
    all_adj.append(adj[0].clone())
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

def diffusion(adj_perturbed, feature_matrix, k):
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

      internal_diffusion = torch.cat(internal_diffusion, dim=0)
      enriched_feature_matrices.append(internal_diffusion)

  feature_matrices_of_perturbations = torch.stack(enriched_feature_matrices)

  return feature_matrices_of_perturbations

class EnrichedGraphDataset(Dataset):
    def __init__(self, root, dataset, k, p, num_perturbations):
        super(EnrichedGraphDataset, self).__init__(root ,transform=None, pre_transform=None)
        self.k = k
        self.p = p
        self.num_perturbations = num_perturbations
        self.data_list = self.process_dataset(dataset)

    def process_dataset(self, dataset):
        #dataset = TUDataset(self.root, name)
        enriched_dataset = []

        for data in dataset:
            edge_index = data.edge_index
            feature_matrix = data.x.clone()

            normalized_adj = compute_symmetric_normalized_adj(edge_index)
            adj = normalized_adj.unsqueeze(0).expand(self.num_perturbations, -1, -1).clone()
            perturbed_adj = generate_perturbation(adj, self.p)
            feature_matrices_of_perts = diffusion(perturbed_adj, feature_matrix, self.k)
            # final_feature_of_graph = feature_matrices_of_perts.mean(dim=0).clone()
            final_feature_of_graph = feature_matrices_of_perts.view(-1, feature_matrices_of_perts.size(-1)).clone()

            enriched_data = Data(x=final_feature_of_graph, edge_index=edge_index, y=data.y)
            enriched_dataset.append(enriched_data)

        return enriched_dataset

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

class DropGNN_V2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
      super(DropGNN_V2, self).__init__()

      self.linear1 = nn.Linear(input_dim, hidden_dim)
      self.relu = nn.ReLU()
      self.linear2 = nn.Linear(hidden_dim, hidden_dim)
      self.linear3 = nn.Linear(hidden_dim, output_dim)
      self.dropout = dropout

      self.reset_parameters()

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear3.reset_parameters()

    def forward(self, data):
      #edge_index = data.edge_index
      x = data.x
      batch = data.batch
      #print(f'x before view: {x.shape}')
      #x = x.view(-1, x.size(-1))
      #print(f'x after view: {x.shape}')
      x = self.relu(self.linear1(x))
      #print(f'x after linear 1: {x.shape}')
      #x = F.dropout(x, p=self.dropout, training=self.training)
      x = self.relu(self.linear2(x))
      #print(f'x after linear 2: {x.shape}')
      x = F.dropout(x, p=self.dropout, training=self.training)

      x = global_add_pool(x, batch)
      #print(f'x after sum pooling: {x.shape}')
      #x = F.dropout(x, p=self.dropout, training=self.training)
      x = self.linear3(x)
      #print(f'x after linear 3: {x.shape}')
      return F.log_softmax(x, dim=-1)

    # def global_sum_pool(self, x, batch):
    #     # x: Input node features of shape (num_nodes, input_dim)
    #     # batch: Batch vector of shape (num_nodes,)
    #     assert x.size(0) == batch.size(0), "Mismatch between x and batch sizes"
    #     # Sum the node features across each graph in the batch
    #     graph_sum = scatter(x, batch, dim=0, reduce='sum')
    #
    #     return graph_sum

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
     return correct / len(loader.dataset)

def main(args):
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
    enriched_dataset = EnrichedGraphDataset(os.path.join(current_path, 'enriched_dataset'), dataset, k=4, p=p,
                                            num_perturbations=num_perturbations)

    seeds_to_test = [0, 64, 1234]
    n_splits = 10
    # test_acc_lists = []
    final_acc = []
    final_std = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    for seed in seeds_to_test:
        print(f'Seed: {seed}')
        print("==============")
        torch.manual_seed(seed)
        np.random.seed(seed)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        best_acc_list = []

        # Iterate through each fold
        for fold, (train_indices, test_indices) in enumerate(kf.split(enriched_dataset)):
            print(f'Fold {fold + 1}/{n_splits}:')

            # Create data loaders for the current fold
            train_loader = DataLoader(
                enriched_dataset[train_indices],
                sampler=RandomSampler(enriched_dataset[train_indices], replacement=True,
                                      num_samples=int(
                                          len(train_indices) * 50 / (len(train_indices) / args.batch_size))),
                batch_size=args.batch_size, drop_last=False,
                collate_fn=Collater(follow_batch=[], exclude_keys=[]))

            test_loader = DataLoader(enriched_dataset[test_indices], batch_size=args.batch_size, shuffle=False)

            # Reinitialize the model for each fold
            model = DropGNN_V2(enriched_dataset.num_features, 64, enriched_dataset.num_classes, args.dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            best_acc = 0
            # test_acc_list = []
            counter = 0

            # Training loop for the current fold
            for epoch in range(1, args.epochs + 1):
                lr = scheduler.optimizer.param_groups[0]['lr']
                train_loss = train(model, train_loader, optimizer, device)
                scheduler.step()
                test_acc = test(model, test_loader, device)
                if epoch % 20 == 0:
                    print(f'Epoch: {epoch:02d} | TrainLoss: {train_loss:.3f} | Test_acc: {test_acc:.3f}')
                # test_acc_list.append(test_acc)

                current_acc = test_acc

                if current_acc > best_acc + args.min_delta:
                    best_acc = current_acc
                    # best_model = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                    counter += 1

                if counter >= args.patience:
                    print(
                        f"Validation performance did not improve by at least {args.min_delta:.3f} for {args.patience} epochs. Stopping training...")
                    print(f"Best validation accuracy: {best_acc:.3f}")
                    print("===============================")
                    break

            # Store the results for the current fold
            best_acc_list.append(best_acc)
            # test_acc_lists.append(test_acc_list)

        # Calculate and report the mean and standard deviation of the best accuracies
        mean_best_acc = np.mean(best_acc_list)
        std_best_acc = np.std(best_acc_list)

        final_acc.append(mean_best_acc)
        final_std.append(std_best_acc)

        print(f'Mean Best Accuracy for seed {seed}: {mean_best_acc}')
        print(f'Standard Deviation for seed {seed}: {std_best_acc}')

    print("======================================")
    print(f'Test accuracy for all the seeds: {np.mean(final_acc)}')
    print(f'Std for all the seeds: {np.mean(final_std)}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['MUTAG', 'IMDB-BINARY', 'IMDB-MULTI'], default='MUTAG',
                        help="Options are ['MUTAG', 'IMDB-BINARY', 'IMDB-MULTI']")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=1234, help='seed for reproducibility')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--dropout', type=float, choices=[0.5, 0.2], default=0.2, help='dropout probability')
    parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')
    parser.add_argument('--min_delta', type=float, default=0.001, help='min_delta in early stopping')
    parser.add_argument('--patience', type=int, default=100, help='patience in early stopping')
    args = parser.parse_args()
    main(args)