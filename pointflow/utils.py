import torch
import os, sys, random, math
from math import log, pi
import pptk 
from PIL import Image
import numpy as np

class dictobj(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

def standard_normal_logp(z):
    """
    compute probability of z under standard normal N(0, I)
    """
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi) # F/2 log(2Pi)
    return log_z - z.pow(2) / 2 # -1/2 z^2 - F/2 log(2Pi) 

def truncated_normal(t: torch.Tensor, mean=0, std=1, truncate_std=2):
    tmp = torch.tensor(t.shape + (4, )).normal_()
    valid = (tmp < truncate_std) & (tmp > -truncate_std) # ? shall it be ``and``
    ind = valid.max(-1, keepdim=True)[1]
    t.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    t.data().mul_(std).add_(mean)

def save(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: str):
    params = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(params, path)

def resume(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer=None, strict=True):
    # ? what is strict param for
    params = torch.load(path)
    model.load_state_dict(params['model'], strict=strict)
    epoch = params['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(params['optimizer'])

    return model, optimizer, epoch

def checkdir(path):
    if os.path.exists(path):
        os.mkdir(path)
        print("Created folder %s" % (path))

class Averager():
    """ An average recorder"""
    def __init__(self):
        self.value, self.average, self.total, self.count = \
            0., 0., 0., 0.
    
    def reset(self):
        self.value, self.average, self.total, self.count = \
            0., 0., 0., 0.

    def update(self, value, n=1):
        self.value = value
        self.count += n
        self.total += n * value
        self.average = self.total / self.count

def visualize(pts, path, samples=1):
    rand = torch.randperm(pts.shape[0])
    _pts = pts[rand][:samples]
    imgs = []
    for p in _pts:
        v = pptk.viewer(pts)
        v.attributes(v)
        v.set(look_at=[1, 1, 1], point_size=0.03, bg_color=[1, 1, 1, 1])
        # save a screenshot
        v.capture(path)
        v.close()
        im = Image.open(path).convert('RGB')
        im = np.array(im)
        im.reshape(1, *im.shape)
        imgs.append(im)
    return np.concatenate(imgs) # NHWC

def apply_random_rotation(pc, rot_axis=1):
    """
    Data Augmentation: random rotation along a axis
    """
    B = pc.shape[0]

    theta = np.random.rand(B) * 2 * np.pi
    zeros = np.zeros(B)
    ones = np.ones(B)
    cos = np.cos(theta)
    sin = np.sin(theta)

    assert rot_axis in [1, 2, 3], 'Rotation Axie must within (1, 2, 3)!'
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

    rot = torch.from_numpy(rot).to(pc)

    # (B, N, 3) mul (B, 3, 3) -> (B, N, 3)
    pc_rotated = torch.bmm(pc, rot)
    return pc_rotated, rot, theta

    

