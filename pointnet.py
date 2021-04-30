import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import torch_geometric as tg
from torch_geometric.data import Data
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader
import os, sys
try:
    import pptk  # the point cloud visualization lib
except:
    print("pptk not installed!")
# from torchsummary import summary
from helpers import *

pl_path = 'data\modelnet40-2500\pointcloud\\'
typenames = (
    'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_training_point_cloud_with_index(ind: int):
    try:
        pptk.viewer(train_tensor[ind].detach().cpu())
    except:
        return None
    # print(typenames[round(train_y[ind].item())])

class TNet(nn.Module):
    """
    In: B*K*N, K=3(Input Trans.), K=64(Feat. Trans.)
    Out : B*K*K
    """
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc = nn.Sequential(nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

        self.k = k

    def forward(self, pc):
        batchsize = pc.shape[0]
        out = self.net1(pc) # shape: B * 1024 * N
        out = out.max(dim=2)[0].view(-1, 1024) # shape: B * 1024 * 1 => B * 1024
        out = self.fc(out) # B * (k ^ 2)
        iden = torch.eye(self.k, device=pc.device, dtype=torch.float32)\
            .view(-1, self.k, self.k)\
            .repeat(batchsize, 1, 1)
        out = out.view(-1, self.k, self.k) + iden
        return out
        # Out: B * K * K

class PointNet(nn.Module):
    """
    Input: B * 3 * N
    Out: B * FEATURE-OUT
    """
    def __init__(self, fout: int):
        super(PointNet, self).__init__()
        self.TNet3d = TNet(k=3)
        self.TNet64d = TNet(k=64)
        self.MLP1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.MLP2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.MLP3 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, fout),
            nn.LogSoftmax(dim=1)  # soft-max within an instance
        )
        self.fout = fout


    def forward(self, pc):
        batchsize = pc.shape[0]
        t_input = self.TNet3d(pc)  # B * 3 * 3
        out = pc
        out = out.transpose(2, 1)  # B * N * 3
        out = torch.bmm(out, t_input)  # B * N * 3
        out = out.transpose(2, 1)  # B * 3 * N
        out = self.MLP1(out)
        out = self.MLP2(out)    # B * 64 * N
        t_feat = self.TNet64d(out)  # B * 64 * 64
        out = out.transpose(2, 1)  # B * N * 64
        out = torch.bmm(out, t_feat)
        out = out.transpose(2, 1)  # B * 64 * N
        out = self.MLP3(out) # B * 1024 * N
        out = out.max(dim=2, keepdim=True)[0].view(-1, 1024) # max-pooling, B * 1024
        out = self.fc(out) # B * FOUT
        return out, t_input, t_feat  # return transforms, to be regularized

class ModelNet10(data.Dataset):
    def __init__(self, shapes, labels):
        super(ModelNet10, self).__init__()
        self.shapes = shapes
        self.labels = labels
        self.length = shapes.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.shapes[item], self.labels[item]


def transformation_feature_regularizer(trans):
    # In: B * K * K
    b, k, _ = trans.shape
    iden = torch.eye(k).view(-1, k, k).repeat(b, 1, 1).to(trans.device)
    loss = torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - iden, dim=(1, 2)).mean()
    return loss

if __name__ == "__main__":
    fout = len(typenames)

    # teset data
    if not os.path.exists(pl_path):
        raise FileNotFoundError('Dataset Tensor Not Found at %s' % (pl_path))

    train_tensor, test_tensor = torch.load(pl_path + 'train.points'), torch.load(pl_path + 'test.points')
    train_y, test_y = torch.load(pl_path + 'train.labels').long(), torch.load(pl_path + 'test.labels').long()
    print(train_tensor.shape) # => 3991 * 2500 * 3

    # model = TNet(k=3).to(device)
    # test model
    model = PointNet(fout=40).to(device) # i.e. 10 for ModelNet-10
    init_weights(model)
    summary(model, input_size=(3, 1024))
    # test output
    train_tensor, train_y = train_tensor.to(device), train_y.to(device)
    idx, bs = 4444, 4
    print(train_y[idx: idx+bs].shape, train_y[idx: idx+bs])
    out, t_in, t_feat = model(train_tensor[idx: idx+bs].transpose(2, 1))
    # print(train_y.shape)
    loss = F.nll_loss(out, train_y[idx: idx+bs].view(-1))
    print(model(train_tensor[idx: idx+bs].transpose(2, 1)), loss)
    # test initial correct rate
    test_dataset = ModelNet10(test_tensor.transpose(2, 1), test_y)
    test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)
    correct = 0
    for i, batch in enumerate(test_loader, 0):
        pcs, labels = batch
        pcs, labels = pcs.to(device), labels.to(device)
        out, t_in, t_feat = model(pcs)
        correct += out.max(dim=1)[1].eq(labels).sum().item()  # count correct samples (in test set)
    t_accu = correct / len(test_dataset)
    print(100 * t_accu)
    show_training_point_cloud_with_index(1222)


