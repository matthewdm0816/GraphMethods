import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.data import Data
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader
import os, sys
from tqdm import tqdm

samplePoints = 1024 # global defined sample points number
typenames = (
    'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'
)
fout = len(typenames)

def normalize(data):
    with torch.no_grad():
        train_min = torch.cat([data.pos[:, i].min().view(1) for i in range(3)])
        res = data.pos - train_min
        train_max = torch.cat([res[:, i].max().view(1) for i in range(3)])
        if train_max.eq(0).sum().item() != 0:
            print(train_max)
        res = res / (train_max + 1e-6) # prevent NaN
        # res = res / train_max
        if torch.isnan(res).sum().item() != 0:
            print("NaN!")
        # print(res.max(), res.min(), torch.isnan(res).sum())
        return res

def scale_normalize(data):
    data = tg.transforms.SamplePoints(1024)(data)
    data = tg.transforms.NormalizeScale()(data)
    return data

if __name__ == "__main__":
    # load ModelNet-10 dataset
    # modelnet = ModelNet(root='data/modelnet10-ori', name='10', train=True, transform=tg.transforms.SamplePoints(samplePoints))
    # test_modelnet = ModelNet(root='data/modelnet10-ori', name='10', train=False, transform=tg.transforms.SamplePoints(samplePoints))

    # load ModelNet-40 dataset
    modelnet = ModelNet(root='F:\\PointNet\\data/modelnet40-%d-normal' % (samplePoints), name='40', train=True,
                        pre_transform=scale_normalize)
    test_modelnet = ModelNet(root='F:\\PointNet\\data/modelnet40-%d-normal' % (samplePoints), name='40',
                             train=False, pre_transform=scale_normalize)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(modelnet, test_modelnet)
    # => ModelNet10(3991) ModelNet10(908)

    # save point cloud formed dataset
    # train = modelnet[0].to(device)
    print('Processing dataset into tensor')
    train_len, test_len = len(modelnet), len(test_modelnet) # 3991 / 9843 || ... / 2468
    train_tensor, test_tensor = torch.zeros([train_len, samplePoints, 3]), torch.zeros([test_len, samplePoints, 3])
    train_y, test_y = torch.zeros([train_len], dtype=torch.int32), torch.zeros([test_len], dtype=torch.int32)
    for i, data in enumerate(modelnet, 0):
        train_tensor[i] = normalize(data) # + torch.randn(data.pos.shape) * 0.02
        train_y[i] = data.y

    for i, data in enumerate(test_modelnet, 0):
        test_tensor[i] = normalize(data)
        test_y[i] = data.y

    pl_path = 'F:\\PointNet\\data/modelnet40-%d-normal/pointcloud/' % (samplePoints)
    if not os.path.exists(pl_path):
        os.makedirs(pl_path)

    print('Saving tensor')
    torch.save(train_tensor, pl_path + 'train.points')
    torch.save(train_y, pl_path + 'train.labels')
    torch.save(test_tensor, pl_path + 'test.points')
    torch.save(test_y, pl_path + 'test.labels')