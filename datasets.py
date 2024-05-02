import torch
from torch_geometric.datasets import TUDataset
import torch.nn.functional as F
import os.path as osp
from ptc_dataset import PTCDataset
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset


class FeatureDegree(BaseTransform):
    r"""Adds the node degree as one hot encodings to the node features.

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """

    def __init__(self, max_degree, in_degree=False, cat=True):
        self.in_degree = in_degree
        self.cat = cat
        self.max_degree = max_degree

    def __call__(self, data):
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = degree(idx, data.num_nodes, dtype=torch.long)
        deg = F.one_hot(deg, num_classes=self.max_degree + 1).to(torch.float)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.long).unsqueeze(-1)
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.in_degree})'


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

    elif 'PTC_GIN' in args.dataset:
        class MyFilter(object):
            def __call__(self, data):
                return True

        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'PTC_GIN')
        dataset = PTCDataset(path, name='PTC')

    elif 'COLLAB' in args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'COLLAB')
        dataset = TUDataset(path, name='COLLAB', transform=FeatureDegree(max_degree=491, cat=False))

    elif 'ogbg-molhiv' in args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ogbg-molhiv')
        dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=path)

    elif 'ogbg-moltox21' in args.dataset:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'ogbg-moltox21')
        dataset = PygGraphPropPredDataset(name="ogbg-moltox21", root=path)

    else:
        raise ValueError("Invalid dataset name")

    return dataset
