import torch
import torch.nn.functional as F
import torch.nn as nn
# from torchsummary import summary
import torch_geometric.nn as tgnn
from torch_geometric.nn import GCNConv, SGConv, MessagePassing
import torch_geometric as tg
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader
import torch_scatter as tscatter
import numpy as np
import random, math


class MLP(nn.Module):
    """
    Plain MLP with activation
    """
    def __init__(self, fin, fout, activation=nn.ReLU, dropout=None, batchnorm=True):
        super().__init__()
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

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super().__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.postfc = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=dropout) if dropout is not None else nn.Identity(),
            nn.ReLU()
        )

    def forward(self, x, w):
        return self.postfc(self.gcn(x, w))

class DenseGCNClassifier(nn.Module):
    """
    GCN, added Dense Links
    """
    def __init__(self, in_channels, classes):
        super().__init__()
        self.gcns = nn.ModuleList([])
        self.sizes = [
            (in_channels, 128), (128 + in_channels, 128), (in_channels + 256, 512), (in_channels + 768, 1024)
        ] 
    
        fc_in = in_channels + 768 + 1024

        for i, o in self.sizes:
            self.gcns.append(GCNLayer(in_channels=i, out_channels=o, dropout=None))

        # for i, gcn in enumerate(self.gcns):
        #     self.register_parameter('dense-gcn%d' % i, gcn)

        self.fc = nn.Sequential(
            MLP(fc_in, 512, dropout=0.3, batchnorm=True, activation=nn.ReLU),
            MLP(512, 128, dropout=0.3, batchnorm=True, activation=nn.ReLU),
            nn.Linear(128, classes),
            nn.LogSoftmax(dim=1)  # soft-max within an instance
        )

    def forward(self, data):
        out, w = data.pos, data.edge_index
        # assert data.batch == data.batch.sort(descending=False, dim=0)[0] # assert batch is sorted along
        # print(data.batch)
        for i, gcn in enumerate(self.gcns):
            _out = gcn(out, w)
            out = torch.cat([out, _out], dim=1)
            # print(out.shape)
        out = tscatter.scatter_max(src=out, index=data.batch, dim=0)[0]  # max-pooling => B * 1024
        out = self.fc(out)  # => B * FOUT
        return out

class GCNClassifierSparse(nn.Module):
    """
    GCN used to classify graphs (point clouds in kNN/rNN sense)
    """
    # TODO: Add Residual/Dense Links
    def __init__(self, in_channels, classes):
        super(GCNClassifierSparse, self).__init__()
        self.gcns = nn.ModuleList([])
        self.sizes = [
            (in_channels, 128), (128, 128), (128, 256), (256, 256), (256, 1024)
        ]
        for i, o in self.sizes:
            self.gcns.append(GCNLayer(in_channels=i, out_channels=o, dropout=None))

        self.fc = nn.Sequential(
            MLP(1024, 512, dropout=0.3, batchnorm=True, activation=nn.ReLU),
            MLP(512, 128, dropout=0.3, batchnorm=True, activation=nn.ReLU),
            nn.Linear(128, classes),
            nn.LogSoftmax(dim=1)  # soft-max within an instance
        )

    def forward(self, data):
        out, w = data.pos, data.edge_index
        for gcn in self.gcns:
            out = gcn(out, w)
        out = tscatter.scatter_max(src=out, index=data.batch, dim=0)[0]  # max-pooling => B * 1024
        out = self.fc(out)  # => B * FOUT
        return out
        
class SGCClassifier(nn.Module):
    """
    Use SGC instead of GCN.
    """
    def __init__(self, in_channels, classes, K=5):
        super().__init__()
        self.sgc, self.br = SGConv(in_channels, 1024, K=K), \
        nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, classes),
            nn.LogSoftmax(dim=1)  # soft-max within an instance
        )
    
    def forward(self, data):
        x, w = data.pos, data.edge_index
        out = self.br(self.sgc(x, w)) # => (B * 1024) * N
        out = tscatter.scatter_max(out, data.batch, dim=0)[0]  # max-pooling => B * 1024
        out = self.fc(out)  # => B * FOUT
        return out

def transform(samplePoints=2500, k=32):
    def f(data):
        data = tg.transforms.NormalizeScale()(data) # normalize to [-1, 1]
        data = tg.transforms.SamplePoints(samplePoints)(data)
        data = tg.transforms.KNNGraph(k=k)(data)
        return data
    return f

def apply_instance_scatter_rotation(pc, batch_size, batch=None, std=1/36):
    orig = pc.size(0)
    n_pts = pc.size(0) // batch_size
    # print(n_pts)
    pc = pc.view(batch_size, n_pts, -1) # => B * N * 3(FIN)

    B = 1
    theta = np.random.rand(B) * 2 * np.pi * std # 1/36 ~ 10-deg by default, at a random axis
    zeros = np.zeros(B)
    ones = np.ones(B)
    cos = np.cos(theta)
    sin = np.sin(theta)

    rot_x = torch.from_numpy(np.stack([
            cos, -sin, zeros,
            sin, cos, zeros,
            zeros, zeros, ones
        ])).view(3, 3)
    rot_y = torch.from_numpy(np.stack([
            cos, zeros, -sin,
            zeros, ones, zeros,
            sin, zeros, cos
        ])).view(3, 3)
    rot_z = torch.from_numpy(np.stack([
            ones, zeros, zeros,
            zeros, cos, -sin,
            zeros, sin, cos
        ])).view(3, 3)

    rots = torch.stack([rot_x, rot_y, rot_z]).to(pc) # => [3, 3, 3]
    axis = torch.randint(0, 3, size=(batch_size,)).view(-1, 1, 1).expand(-1, 3, 3).to(pc) # => [B * 3 * 3]
    rots = torch.gather(input=rots, dim=0, index=axis.long()).to(pc) # => [B, 3, 3]

    pc_rotated = torch.bmm(pc, rots).view(orig, -1)
    return pc_rotated, rots


def apply_random_rotation(pc, rot_axis=1, std=1. ):
    """
    Data Augmentation: random rotation along a axis
    Batched version
    """
    B = 1

    theta = np.random.rand(B) * 2 * np.pi * std
    zeros = np.zeros(B)
    ones = np.ones(B)
    cos = np.cos(theta)
    sin = np.sin(theta)

    assert rot_axis in [1, 2, 3], 'Rotation Axis must within (1, 2, 3)!'
    if rot_axis == 0:
        rot = np.stack([
            cos, -sin, zeros,
            sin, cos, zeros,
            zeros, zeros, ones
        ]).T.reshape(B, 3, 3)
    elif rot_axis == 1:
        rot = np.stack([
            cos, zeros, -sin,
            zeros, ones, zeros,
            sin, zeros, cos
        ]).T.reshape(B, 3, 3)
    else:
        rot = np.stack([
            ones, zeros, zeros,
            zeros, cos, -sin,
            zeros, sin, cos
        ]).T.reshape(B, 3, 3)

    rot = torch.from_numpy(rot).to(pc).view(3, 3)

    # (B * N, 3) mul (3, 3) -> (B * N, 3)
    pc_rotated = torch.mm(pc, rot)
    return pc_rotated, rot, theta

def apply_random_scale(pc, batch_size, std=.1):
    orig = pc.shape[0]
    n_pts = orig // batch_size
    B = batch_size

    scales = torch.randn((B, 3)) * std + 1.
    scale_mat = torch.diag_embed(scales).to(pc) # => (B * 3 * 3)

    pc_scaled = torch.bmm(pc.view(B, n_pts, -1), scale_mat).view(orig, -1)

    return pc_scaled, scale_mat

def apply_random_flip(pc, flip_axis=1):
    assert flip_axis in [1, 2, 3], 'Flip Axis must within (1, 2, 3)!'
    # pc: (B * N) * 3
    flip = torch.ones((3,)).to(pc)
    flip[flip_axis - 1] = -1
    # if flip_axis == 1:
    #     flip[0] = -1
    # elif flip_axis == 2:
    #     flip[1] = -1
    # elif flip_axis == 3:
    #     flip[2] = -1
    pc_flip = pc * flip
    return pc, flip

def apply_scatter_random_flip(pc, batch_size=1):
    axis = torch.from_numpy(np.random.choice(a=[0, 1, 2], size=batch_size))
    axis = -2 * F.one_hot(axis, num_classes=3) + 1 # B * 3
    pc_bs = pc.view(batch_size, -1, 3)
    
    flip_mat = torch.diag_embed(axis).to(pc) # => (B * 3 * 3)
    pc_bs = torch.bmm(pc_bs, flip_mat).view(-1, 3)
    return pc_bs, axis


def apply_random_pertubation(pc, mean=None, std=0.003):
    """
    Random Pertubation Augmentation | 1% by default
    """
    scale = pc.max(dim=0)[0] - pc.min(dim=0)[0]
    std = std * scale
    mean = pc.mean()
    D = pc.shape[0]
    
    pertubation = torch.randn((D, 3)).to(pc) * std + mean
    return pc + pertubation, pertubation
    
def cal_loss(pred, gold, smoothing=True, label_weights=None, gamma=1.5):
    ''' 
    Calculate cross entropy loss, apply label smoothing if needed. 
    Modified from DGCNN original implementation
    Input changed to log-softmaxed logits.
    Focal Loss Added
    '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        # one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = F.one_hot(gold, num_classes=n_class).to(pred)
        one_hot_smooth = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1) # (e/(n-1), ..., 1-e, ..., e/(n-1)) smooth labels
        log_prb = pred
        if label_weights is not None:
            # label_weights = label_weights.view(1, -1)   # => (1, C)
            w = torch.sum(label_weights * one_hot, dim=1).mean()  
        else:
            w = torch.tensor(1.)
        focal_weight = torch.pow(1 - log_prb, gamma)

        loss = -w * (one_hot_smooth * log_prb) * focal_weight
        loss = loss.sum() / (focal_weight.mean(dim=1).sum())# TODO: Maybe SUM is better?
    else:
        loss = F.nll_loss(pred, gold, reduction='mean')

    return loss

if __name__ == '__main__':
    samplePoints = 1024
    batch_size = 16
    # n_points = 320
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCNClassifier(in_channels=3, classes=10).to(device)
    # summary(model, input_size=[(3, n_points), (n_points, n_points)])
    # del model
    # model_geo = GCNClassifierSparse(in_channels=3, classes=10).to(device)
    # modelnet = ModelNet(root='data/modelnet10', name='10', train=True,
    #                     transform=transform)
    # loader = DataLoader(modelnet, batch_size=24, shuffle=True)
    # for i, batch in enumerate(loader, 0):
    #     out = model_geo(batch.to(device))
    #     print(out, out.shape, sep='\n')
    #     break
    # del model_geo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GCNClassifierSparse(in_channels=3, classes=40).to(device)  # train on ModelNet-40
    train_dataset = ModelNet(root='data/modelnet40', name='40', train=True,
                            pre_transform=tg.transforms.SamplePoints(samplePoints),
                            transform=tg.transforms.KNNGraph(k=10))
    test_dataset = ModelNet(root='data/modelnet40', name='40', train=False,
                            pre_transform=tg.transforms.SamplePoints(samplePoints),
                            transform=tg.transforms.KNNGraph(k=10))
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# %%
