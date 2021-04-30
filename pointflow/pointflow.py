import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchdiffeq as tde 

from model import get_point_cnf, get_latent_cnf
from utils import truncated_normal, standard_normal_logp, dictobj, visualize

import math, os, sys, random

"""
Basic Architecture: diffop => odenet, cnf => model => pointflow => train
"""

class PNMLP(nn.Module):
    """
    PointNet-like MLP/1d-Conv
    """
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            activation() if activation is not None else nn.Identity()
        )
        
    def forward(self, x):
        return self.net(x)


class BRMLP(nn.Module):
    """
    BatchNormed MLP
    """
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            activation() if activation is not None else nn.Identity()
        )
        
    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, zdim, in_channels=3, deterministic=False):
        super().__init__()
        self.deterministic = deterministic
        self.convs = nn.ModuleList([])

        self.sizes = [(in_channels, 128), (128, 128), (128, 256), (256, 512)]
        self.fcsizes = [(512, 256), (256, 128), (128, zdim)]

        for index, (i, o) in enumerate(self.sizes):
            self.convs.append(PNMLP(i, o, activation=nn.ReLU if index != len(self.sizes)-1 else None))

        if self.deterministic: # use deterministic encoder
            self.fc = nn.ModuleList([])
            for index, (i, o) in enumerate(self.fcsizes):
                self.fc.append(BRMLP(i, o, activation=nn.ReLU if index != len(self.sizes)-1 else None))

        else: # use variational encoder, encode to mean + logvar
            self.fc_mean = nn.ModuleList([])
            self.fc_var = nn.ModuleList([])
            for index, (i, o) in enumerate(self.fcsizes):
                self.fc_mean.append(BRMLP(i, o, activation=nn.ReLU if index != len(self.sizes)-1 else None))
                self.fc_var.append(BRMLP(i, o, activation=nn.ReLU if index != len(self.sizes)-1 else None))

    def forward(self, x):
        x = x.transpose(2, 1) # B * N * 3 => B * 3 * N
        for conv in self.convs:
            x = conv(x)
        
        x = torch.max(x, dim=2)[0] # B * 512 * 1 ?
        m, v = x, x
        if self.deterministic:
            for mlp in self.fc:
                m = mlp(m)
            v = 0
        else:
            for mlp_mean, mlp_var in zip(self.fc_mean, self.fc_var):
                m, v = mlp_mean(m), mlp_var(v)
        
        return m, v
            
class PointFlow(nn.Module):
    """
    Basic Architeture:
    X(point clouds) 🢣Encoder🢣 Z(latent repr.) 🢣Latent CNF $F$🢣 W(latent prior, Gaussian) 
                                    ▨▨⟱⟱▨▨
                            X    🢣Point CNF $G$🢣 Y(p.c. prior, Gaussian)
    """
    def __init__(self, **args):
        super().__init__()
        args = dictobj(args)
        self.fin, self.fz = args.fin, args.fz
        self.use_latent_flow, self.use_deterministic_encoder = args.use_latent_flow, args.use_deterministic_encoder
        self.prior_w, self.recon_w, self.entropy_w = args.prior_w, args.recon_w, args.entropy_w
        self.distributed = args.distributed
        self.encoder = Encoder(zdim=args.fz, in_channels=args.fin, deterministic=self.use_deterministic_encoder)
        self.pointCNF = get_point_cnf(args)
        self.latentCNF = get_latent_cnf(args) if self.use_latent_flow else nn.Identity()

    @staticmethod
    def reparametrized_gaussian(mean, logvar):
        """
        generate a $N(\\mu, \\sigma)$ using reparam.
        """
        std = torch.exp(logvar * 0.5)
        eps = std * torch.randn(std.shape).to(mean)
        return mean + eps

    @staticmethod
    def sample_gaussian(shape, truncate_std=None, gpu=None):
        """
        sample a gaussian(noise), capable to use truncated normal dist.
        """
        y = torch.randn(shape)
        y = y.cuda(gpu) if gpu is not None else y
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, truncate_std=truncate_std)
        return y

    @staticmethod
    def gaussian_entropy(logvar):
        const = 0.5 * float(logvar.size(1)) * (1. + torch.log(2 * np.pi)) 
            # ? I'm wondering, (including in VAEs) is this constant term neccesary?
        ent = 0.5 * logvar.sum(dim=1, keepdim=False)
        return const + ent

    def multigpu_wrapper(self, f):
        raise NotImplementedError()

    def get_optimizer(self, **args):
        args = dictobj(args)
        def _get_optimizer(params):
            if args.optimizer == optim.Adam: 
                opt = optim.Adam(params, 
                    lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
            elif args.optimizer == optim.SGD: # ? Maybe try Nesterov optimizer
                opt = optim.SGD(params, 
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                raise Exception("Invalid Optimizer Type, should be either optim.Adam or optim.SGD")

            return opt
        opt = _get_optimizer(list(self.pointCNF.parameters()) +
                            list(self.latentCNF.parameters()) +
                            list(self.encoder.parameters())
                            )
        return opt

    def encode(self, x):
        z_mu, z_logvar = self.encoder(x)
        if self.use_deterministic_encoder:
            return z_mu
        else:
            return self.reparametrized_gaussian(z_mu, z_logvar)

    def decode(self, z, num_pts, truncate_std=None):
        y = self.sample_gaussian((z.size(0), num_pts, self.fin), truncate_std=truncate_std)
            # z.size(0) ~ batch_size
        x_recon = self.pointCNF(y, z, reverse=True).view(*y.size())
        return y, x_recon

    def sample(self, batch_size, num_pts, truncate_std=None, truncate_std_latent=None, gpu=None):
        assert self.use_latent_flow
        w = self.sample_gaussian(shape=(batch_size, self.fz), truncate_std=truncate_std_latent, gpu=gpu)
            # note Z, W is of same dimesionaliy
        z = self.latentCNF(w, None, reverse=True).view(*w.shape)
        y = self.sample_gaussian(shape=(batch_size, num_pts, self.fin), truncate_std=truncate_std, gpu=gpu)
        x_recon = self.pointCNF(y, z, reverse=True).view(*y.shape)
        return z, x_recon

    def reconstruct(self, x, num_pts=None, truncate_std=None):
        num_pts = num_pts if num_pts is not None else x.size(1) # defaultly use input's number of points
        z = self.encode(x)
        _, x = self.decode(z, num_pts, truncate_std)
        return x

    def forward(self, x, opt: optim.Optimizer, step, summary_writer: torch.utils.tensorboard.SummaryWriter=None, sample_gpu=None):
        """
        train inside forward
        """
        opt.zero_grad()
        batch_size, num_pts = x.shape[:2]
        z_mu, z_sigma = self.encoder(x)
        # Compute Q(z|X) and entropy H{Q(z|X)}
        if self.use_deterministic_encoder:
            z = z_mu + 0 * z_sigma # ? why, the original code added this 0 multiplier
            entropy = torch.zeros(batch_size).to(z)
        else:
            z = self.reparametrized_gaussian(z_mu, z_sigma)
            entropy = self.gaussian_entropy(z_sigma)

        # Compute prior P(z)
        if self.use_latent_flow:
            w, dlog_pw = self.latentCNF(z, None, torch.zeros(batch_size, 1).to(z))
            log_pw = standard_normal_logp(w).view(batch_size, -1).sum(dim=1, keepdim=True)
            dlog_pw = dlog_pw.view(batch_size, 1).to(z)
            log_pz = log_pw - dlog_pw
        else:
            log_pz = torch.zeros(batch_size, 1).to(z)

        # Compute recon. P(X|z)
        z_new = z.view(z.shape) + (log_pz * 0.).mean() # ? why
        y, dlog_py = self.pointCNF(x, z_new, torch.zeros(batch_size, num_pts, 1).to(x))
        log_py = standard_normal_logp(y).view(batch_size, -1).sum(dim=1, keepdim=True) 
        dlog_py = dlog_py.view(batch_size, num_pts, 1).to(x)
        log_px = log_py - dlog_py

        # Loss
        entropy_loss = -entropy.mean() * self.entropy_w
        recon_loss = -log_px.mean() * self.recon_w
        prior_loss = -log_pz.mean() * self.prior_w
        loss = entropy_loss + recon_loss + prior_loss
        loss.backward()
        opt.step()

        # Write logs
        if self.distributed: 
            raise NotImplementedError("Distributed training not implemented!")
        else:
            entropy_log = entropy.mean()
            recon_log = -log_px.mean()
            prior_log = -log_pz.mean()

        recon_nats = recon_log / float(x.size(1) * x.size(2))
        prior_nats = prior_log / float(self.fz)

        # reconstruct to save
        with torch.no_grad():
            recon_pc = self.reconstruct(x, truncate_std=True)
            recon_im = visualize(recon_pc, path='/home/tmp/screenshot.png', samples=1)
            
        # sample to save
        if self.use_latent_flow:
            with torch.no_grad():
                sample_pc = self.sample(1, 1024, gpu=sample_gpu)
                sample_im = visualize(sample_pc, samples=1, path='/home/tmp/screenshot.png')
                

        record_dict = {
            'train/entropy': entropy_log.cpu().detach().item() if not isinstance(entropy_log, float) else entropy_log,
            'train/prior': prior_log,
            'train/recon': recon_log,
            'train/recon-nats': recon_nats,
            'train/prior-nats': prior_nats,
            # 'train/sample-reconstructed': recon_pc
        }

        if summary_writer is not None:
            for key, value in record_dict:
                summary_writer.add_scalar(key, value, step)

        record_dict['train/sample-reconstructed'] = recon_im
        summary_writer.add_images('train/sample-reconstructed', recon_im, step, dataformats='NHWC')
        record_dict['train/sample-sampled'] = sample_im
        summary_writer.add_images('train/sample-sampled', sample_im, step, dataformats='NHWC')
        return record_dict





            
        
