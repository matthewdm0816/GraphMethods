import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
# from torchsummary import summary
import random, math, os, sys


def get_distance_matrix(pcs):
    """
    TC: O(BdN^2)
    """
    dims = len(pcs.shape)
    if dims == 2:
        pcs = pcs.unsqueeze(dim=0)
    b, n, d = pcs.shape

    dmat = pcs.view(-1, 1, n, d).repeat(1, n, 1, 1)
    dmat = dmat - dmat.transpose(2, 1)
    dmat = dmat.norm(dim=3)  # => B * N * N
    return dmat if dims == 3 else torch.squeeze(dmat, dim=0)


def make_kNN_graph(pcs, k=10):
    """
    In: B * N * d, for 3D PC, d=3.
    Out: B * N * N, adjacent matrix for each PC.
    k ~ nearest point
    """
    b, n, d = pcs.shape
    dmat = get_distance_matrix(pcs)
    val, _ = torch.kthvalue(dmat, k + 1, dim=2 ,keepdim=True)
    dmat = dmat.le(val).float()
    iden = torch.eye(n).view(1, n, n).repeat(b, 1, 1)
    return dmat - iden  # reduce self-loops


def make_rNN_graph(pcs, r=0.1):
    """
    In: B * N * d, for 3D PC, d=3.
    Out: B * N * N, adjacent matrix for each PC.
    r ~ radius of ball, whole space normalized to 1
    """
    b, n, d = pcs.shape
    dmat = get_distance_matrix(pcs)
    dmat = dmat.le(r).float()
    iden = torch.eye(n).view(1, n, n).repeat(b, 1, 1)
    return dmat - iden  # reduce self-loops


def fps_point_kNN_patch(pcs, samples=10, k=10):
    """
    In: B * N * d, for 3D PC, d=3.
    Out: B * S * k * d(features) + B * S(* 1)(positions), S ~ samples count
    Perform Farthest Point Sampling(FPS)
    TC: O(BdSk N)
    ! Use under torch.no_grad()
    ! Note origin point of each patch is not included in each feature
    """
    b, n, d = pcs.shape
    result, position = torch.zeros((b, samples, k, d)), torch.zeros((b, samples))
    # current_idx = random.choice(range(n))
    for ib, pc in enumerate(pcs, 0):    # maybe we have to do in pc-wise, can't store all dist. mat.
        dmat = get_distance_matrix(torch.unsqueeze(pc, dim=0))[0]
        val, _ = torch.kthvalue(dmat, k + 1, dim=1, keepdim=True)
        mask = dmat.le(val)
        mask ^= torch.eye(n).bool()  # remove self-loops
        # print(mask.sum(dim=1))
        current_idx = random.choice(range(n))  # initial sample
        for i in range(samples):
            if i != 0:
                current_idx = dmat[current_idx].argmax()  # sample fp
            position[ib][i] = current_idx
            result[ib][i] = pc[mask[current_idx]] - pc[current_idx]

    return result, position




def make_centered_kNN_patches(pcs: torch.FloatTensor, idx: torch.LongTensor, k=10):
    """
    INPUT: B * N * D :: B * T ~ T patches to take
    Output: B * T * K * D
    """
    b, n, d = pcs.shape
    dmat = get_distance_matrix(pcs)
    val, _ = torch.kthvalue(dmat, k + 1, dim=2, keepdim=True)

    pass



def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def check_dir(path, verbose=True, color=""):
    if not os.path.exists(path):
        os.makedirs(path)
        if verbose:
            print(color + "Creating folder %s" % path)
    elif verbose:
        print(color + "Folder %s exists" % path)


def gnn_model_summary(model):
    model_params_list = list(model.named_parameters())
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("----------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>20}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("----------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)


if __name__ == '__main__':
    a = torch.randn((1, 8, 3))
    a[0][1] = torch.tensor([0, 0, 0])
    a[0][0] = torch.tensor([0, 0, 0.02])
    a[0][2] = torch.tensor([0, 0, 0.01])
    g1 = make_kNN_graph(a, k=3)
    g2 = make_rNN_graph(a, r=0.1)

    print(g1[0], g2[0], sep='\n')

