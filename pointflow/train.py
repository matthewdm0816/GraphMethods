import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from pointflow import PointFlow
from utils import dictobj, save, resume, visualize, Averager, apply_random_rotation
from datasets import get_datasets
import diffop

import time, random
from tqdm import tqdm, trange

def main():
    torch.backends.cudnn.benchmark = True

    # hyper-params initializing
    args = dictobj()
    args.gpu = torch.device('cuda:%d' % (6)) 
    timestamp = '%d-%d-%d-%d-%d-%d-%d-%d-%d' % time.localtime(time.time())
    args.log_name = '%s-pointflow' % timestamp
    writer = SummaryWriter(comment=args.log_name)
    
    args.use_latent_flow, args.prior_w, args.entropy_w, args.recon_w = True, 1., 1., 1.
    args.fin, args.fz = 3, 128
    args.use_deterministic_encoder = True
    args.distributed = False
    args.optimizer = optim.Adam
    args.batch_size = 16
    args.lr, args.beta1, args.beta2, args.weight_decay = 1e-3, 0.9, 0.999, 1e-4
    args.T, args.train_T, args.atol, args.rtol = 1., False, 1e-5, 1e-5
    args.layer_type = diffop.CoScaleLinear
    args.solver = 'dopri5'
    args.use_adjoint, args.bn = True, False
    args.dims, args.num_blocks = (512, 512), 1 # originally (512 * 3)
    args.latent_dims, args.latent_num_blocks = (256, 256), 1

    args.resume, args.resume_path = False, None
    args.end_epoch = 2000
    args.scheduler, args.scheduler_step_size = optim.lr_scheduler.StepLR, 20
    args.random_rotation = True
    args.save_freq = 10

    args.dataset_type = 'shapenet15k'
    args.cates = ['airplane'] # 'all' for all categories training
    args.tr_max_sample_points, args.te_max_sample_points = 2048, 2048
    args.dataset_scale = 1.0
    args.normalize_per_shape = False
    args.normalize_std_per_axis = False
    args.num_workers = 4
    args.data_dir = "/data/ShapeNetCore.v2.PC15k"


    torch.cuda.set_device(args.gpu)
    model = PointFlow(**args).cuda(args.gpu)

    # load milestone
    epoch = 0
    optimizer = model.get_optimizer(**args)
    if args.resume:
        model, optimizer, epoch = resume(
            args.resume_path, model, optimizer, strict=True
        )
        print("Loaded model from %s" % args.resume_path)

    # load data
    train_dataset, test_dataset = get_datasets(args)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, sampler=None, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
        pin_memory=True, sampler=None, drop_last=False
    )

    if args.scheduler == optim.lr_scheduler.StepLR:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=0.65)
    else:
        raise NotImplementedError("Only StepLR supported")

    ent_rec, latent_rec, recon_rec = Averager(), Averager(), Averager()
    for e in trange(epoch, args.end_epoch):
        # record lr
        if writer is not None:
            writer.add_scalar('lr/optimizer', scheduler.get_lr()[0], e)

        # feed a batch, train
        for idx, data in enumerate(tqdm(train_loader)):
            idx_batch, tr_batch, te_batch = data['idx'], data['train_points'], data['test_points']
            model.train()
            if args.random_rotation:
                # raise NotImplementedError('Random Rotation Augmentation not implemented yet')
                tr_batch, _, _ = apply_random_rotation(
                    tr_batch, rot_axis=train_loader.dataset.gravity_axis
                )
            inputs = tr_batch.cuda(args.gpu, non_blocking=True)
            step = idx + len(train_loader) * e # batch step
            out = model(inputs, optimizer, step, writer, sample_gpu=args.gpu)
            entropy, prior_nats, recon_nats = out['entropy'], out['prior_nats'], out['recon_nats']
            ent_rec.update(entropy)
            recon_rec.update(recon_nats)
            latent_rec.update(prior_nats)

        # update lr
        scheduler.step(epoch=e)

        # save milestones
        if e % args.save_freq == 0 and e != 0:
            save(model, optimizer, e, path='milestone-%d.save' % e)
            save(model, optimizer, e, path='milestone-latest.save' % e) # save as latest model

        
if __name__ == '__main__':
    main()






