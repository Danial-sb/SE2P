import os
import random
import time

import numpy as np
import torch
import wandb

from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

from ogb.graphproppred import Evaluator

from SE2P import EnrichedGraphDataset, SE2P_C1, SE2P_C2, SE2P_C3, SE2P_C4, count_parameters
from DROPGNN import GCN, DropGCN, GIN, DropGIN
from datasets import get_dataset

from args import get_args
from SE2P import sweep_id, sweep_config


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
    args = get_args()

    if args.dataset not in ['ogbg-molhiv', 'ogbg-moltox21']:
        raise ValueError("Invalid dataset")

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
    # num_perturbations = round(gamma * np.log10(gamma)) # Commented out based on DropGNN code.
    num_perturbations = gamma
    print(f'Number of perturbations: {num_perturbations}')
    print(f'Sampling probability: {p}')
    print(f'Number of features: {dataset.num_features}')

    current_path = os.getcwd()
    seeds_to_test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    all_validation_acc = []
    all_test_acc = []

    with wandb.init(config=config):
        config = wandb.config
        for seed in seeds_to_test:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            if args.configuration == 'c1' or args.configuration == 'c2' or args.configuration == 'c3' or args.configuration == 'c4':
                name = f"enriched_{args.dataset}_{args.configuration}"
                start_time = time.time()
                enriched_dataset = EnrichedGraphDataset(os.path.join(current_path, 'enriched_dataset'), name, dataset,
                                                        p=p, num_perturbations=num_perturbations, args=args)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Done! Time taken: {elapsed_time:.2f} seconds")
                print(f'Number of enriched features: {enriched_dataset.num_features}')

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # device = torch.device('cpu')
            print(f'Device: {device}')
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            split_idx = dataset.get_idx_split()
            if args.configuration == 'c1' or args.configuration == 'c2' or args.configuration == 'c3' or args.configuration == 'c4':
                train_loader = DataLoader(enriched_dataset[split_idx["train"]], batch_size=config.batch_size,
                                          shuffle=True)
                valid_loader = DataLoader(enriched_dataset[split_idx["valid"]], batch_size=config.batch_size,
                                          shuffle=False)
                test_loader = DataLoader(enriched_dataset[split_idx["test"]], batch_size=config.batch_size,
                                         shuffle=False)
            else:
                train_loader = DataLoader(dataset[split_idx["train"]], batch_size=config.batch_size, shuffle=True)
                valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=config.batch_size, shuffle=False)
                test_loader = DataLoader(dataset[split_idx["test"]], batch_size=config.batch_size, shuffle=False)

            if args.configuration == 'c1':
                model = SE2P_C1(enriched_dataset.num_features, 1 if args.dataset == 'ogbg-molhiv' else 12, args,
                                config).to(device)

            elif args.configuration == 'c2':
                model = SE2P_C2(enriched_dataset.num_features, 1 if args.dataset == 'ogbg-molhiv' else 12, args).to(
                    device)

            elif args.configuration == 'c3':
                model = SE2P_C3(enriched_dataset.num_features, 1 if args.dataset == 'ogbg-molhiv' else 12,
                                num_perturbations, device, args).to(device)

            elif args.configuration == 'c4':
                model = SE2P_C4(enriched_dataset.num_features, 1 if args.dataset == 'ogbg-molhiv' else 12,
                                num_perturbations, device, args).to(device)

            elif args.configuration == 'GIN':
                model = GIN(dataset.num_features, 1 if args.dataset == 'ogbg-molhiv' else 12, args).to(device)

            elif args.configuration == 'GCN':
                model = GCN(dataset.num_features, 1 if args.dataset == 'ogbg-molhiv' else 12, args).to(device)

            elif args.configuration == 'DropGIN':
                model = DropGIN(dataset.num_features, 1 if args.dataset == 'ogbg-molhiv' else 12, num_perturbations, p,
                                args).to(device)

            elif args.configuration == 'DropGCN':
                model = DropGCN(dataset.num_features, 1 if args.dataset == 'ogbg-molhiv' else 12, num_perturbations, p,
                                args).to(device)

            else:
                raise ValueError("Invalid model name")

            evaluator = Evaluator(name=args.dataset)

            print(f'Model learnable parameters for {model.__class__.__name__}: {count_parameters(model)}')
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

            start_outer = time.time()
            best_val_perf = test_perf = float('-inf')
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

                time_per_epoch = time.time() - start

                if epoch % 25 == 0 or epoch == 1:
                    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                          f'Val: {val_perf:.4f}, Test: {test_perf:.4f}, Seconds: {time_per_epoch:.4f},'
                          f' Memory allocated: {memory_allocated}, Memory Reserved: {memory_reserved}')

            time_average_epoch = time.time() - start_outer
            print(
                f'Best Validation in seed {seed}: {best_val_perf}, Test in seed {seed}: {test_perf}, Seconds/epoch: {time_average_epoch / args.epochs},'
                f' Max memory allocated: {max_memory_allocated}, Max memory reserved: {max_memory_reserved}')
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

        wandb.log({
            "Best validation in all the seeds": best_epoch_mean_val,
            "Std validation in all the seeds": best_epoch_std_val,
            "Best test in all the seeds": best_epoch_mean_test,
            "Std test in all the seeds": best_epoch_std_test
        })


if __name__ == "__main__":
    wandb.agent(sweep_id, main)
