"""
Model Definitions of PointNet++
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import torch_geometric as tg
from torch_geometric.data import Data
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader
import os, sys
import pptk  # the point cloud visualization lib.
from torchsummary import summary
from helpers import *
from milestone import MilestoneRecorder

from pointnet import *

class PointNetPP(nn.Module):
    def __init__(self, fout=10, samples=100, k=16):
        super(PointNetPP, self).__init__()
        self.fout = fout
        self.samples = samples
        self.k = k
        self.pn1 = PointNet(fout=256)
        self.pn2 = PointNet(fout=256)


    def forward(self, pcs):
        out = self.pn1(pcs)

        pass






if __name__ == '__main__':
    # recorder = MilestoneRecorder(comment='pointnetpp')
    # writer =
