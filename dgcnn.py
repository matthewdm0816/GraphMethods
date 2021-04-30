#!/usr/bin/env python
# coding: utf-8

# In[81]:


import torch
import torch.nn.functional as F
import torch.nn as nn
try:
    from torchsummary import summary
except:
    summary = None
from torch_geometric.nn import GCNConv, EdgeConv, DynamicEdgeConv
import torch_geometric.nn as tgnn
import torch_geometric as tg
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader
import torch_scatter as tscatter
import os, sys
from collections import namedtuple
from helpers import gnn_model_summary

class MLP(nn.Module):
    """
    Plain MLP with activation
    """
    def __init__(self, fin, fout, activation=nn.ReLU, dropout=None, batchnorm=True):
        super(MLP, self).__init__()
        if dropout is not None and batchnorm:
            assert isinstance(dropout, float)
            self.net = nn.Sequential(
                nn.Linear(fin, fout),
                nn.BatchNorm1d(fout),
                nn.Dropout(p=dropout),
                activation()
            )
        elif batchnorm:
            self.net = nn.Sequential(
                nn.Linear(fin, fout),
                nn.BatchNorm1d(fout),
                activation()
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(fin, fout),
                activation()
            )

    def forward(self, x):
        return self.net(x)


class MLP_EdgeConv(nn.Module):
    """
    MLP used in aggregation/EdgeConv Layer.
    """
    def __init__(self, in_channels, out_channels, hidden=512):
        super(MLP_EdgeConv, self).__init__()
        self.mlp = nn.Sequential(
            MLP(in_channels, hidden),
            # MLP(1024, 512, dropout=0.3),
            MLP(hidden, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)


class DGCNNClassifier(nn.Module):
    """
    DGCNN used to classify graphs (point clouds in kNN/rNN sense)
    """

    def __init__(self, in_channels, classes):
        super(DGCNNClassifier, self).__init__()
        self.align =  nn.Linear(in_channels, in_channels)
        self.ec1 = DynamicEdgeConv(MLP_EdgeConv(2 * 3, 64, hidden=128), k=16, aggr='max')
        self.ec2 = DynamicEdgeConv(MLP_EdgeConv(2 * 64, 64, hidden=128), k=16, aggr='max')
        self.ec3 = DynamicEdgeConv(MLP_EdgeConv(2 * 64, 128, hidden=256), k=16, aggr='max')
        self.mlp =  MLP(256, 1024) # fin = (128+64+64)=256
        self.fc = nn.Sequential(
            MLP(1024, 512),
            MLP(512, 256),
            MLP(256, classes, batchnorm=False, dropout=0.3, activation=nn.LogSoftmax)
        )

    def forward(self, data):
        x, batch = data.pos, data.batch
        out = self.align(x)     # (B * N) * 3 => (B * N) * 3
        out1 = self.ec1(out, batch)    # (B * N) * 64
        out2 = self.ec2(out1, batch)
        out3 = self.ec3(out2, batch)
        out = torch.cat((out1, out2, out3), dim=1)  # (B * N) * 256
        out = self.mlp(out)
        out = tscatter.scatter_max(out, batch, dim=0)[0] # (B) * 1024, max-pooling
        # max-pooling
        out = self.fc(out)
        return out


class DGCNNTest(DGCNNClassifier):
    def __init__(self, batch, *args, **kwargs):
        super(DGCNNTest, self).__init__(*args, **kwargs)
        self.batch = batch

    def forward(self, x):
        l = x.shape[0]
        Batch = namedtuple('Batch', ['pos', 'batch'])
        b = Batch(pos=x.squeeze(0), batch=torch.zeros(l).long())
        return super().forward(b)

if __name__ == '__main__':

    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

    dataset_type = '40'
    assert dataset_type in ['10', '40']
    if dataset_type == '10':
        pl_path = 'modelnet-10-pointcloud'
        model_path = 'modelnet10-pointnet'
        fout = 10
    elif dataset_type == '40':
        pl_path = 'modelnet40-2500'
        model_path = 'modelnet40-pointnet'
        fout = 40
    assert pl_path and model_path and fout

    model = DGCNNClassifier(in_channels=3, classes=40).to(device)
    train_dataset = ModelNet(root=os.path.join('data',pl_path), name='40', train=True,
                             # pre_transform=tg.transforms.SamplePoints(samplePoints),
                             # transform=tg.transforms.KNNGraph(k=10))
                             pre_transform=tg.transforms.SamplePoints(2500))
    test_dataset = ModelNet(root=os.path.join('data',pl_path), name='40', train=False,
                            # pre_transform=tg.transforms.SamplePoints(samplePoints),
                            # transform=tg.transforms.KNNGraph(k=10))
                            pre_transform=tg.transforms.SamplePoints(2500))

    loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    for b in loader:
        x, batch = b.pos, b.batch
        # if summary is not None:
        #     model2 = DGCNNTest(in_channels=3, classes=40, batch=batch).to(device)
        #     summary(model2, input_size=(1024*1, 3))
        gnn_model_summary(model)
        print(x, batch)
        print(model(x, batch))
        break
    # TODO: Test Code





