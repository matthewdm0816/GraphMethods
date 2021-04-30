import torch
import torch.optim as optim
from gcn import *
from tqdm import  *
from helpers import *
# from train_pointnet import save_model, init_weights
from tensorboardX import SummaryWriter
import time, random, os, sys, gc, copy, colorama

from torch_geometric.nn import DataParallel
from torch_geometric.data import DataListLoader

import numpy as np

dataset_type = '40'
samplePoints = 1024
# batch_size = 32  # largest affordable on GTX1080 with 8G VRAM quq, even with sparse
epochs = 1001
milestone_period = 10

gpu_id = 0
# gpu_ids = [0, 1, 2, 7]
gpu_ids = [0, 1]
ngpu = len(gpu_ids)
# os.environ['CUDA_VISIBLE_DEVICES'] = repr(gpu_ids)[1:-1]
parallel = (ngpu > 1) 
assert gpu_id in gpu_ids

use_unbal_lw = True
model_type = 'dense-gcn'
data_load_only = False
random_rotation = False
random_pertubation = True
use_smooth_loss = True
instance_rotation = True
instance_scale = True
use_ensemble_test = True
random_flip = True
use_sbn = True and parallel # need to use parallel first
rescale_ensemble = True

colorama.init(autoreset=True)

model_milestone, optim_milestone, beg_epochs = \
    '/data/pkurei/PointNet/model/modelnet40-dense-gcn-3%%25noise-smooth-label-weighted-instance-rot-10deg-10%%25rescale-ensemble20/2020-12-11-15-20-26-4-346-0/model-latest.save', \
    '/data/pkurei/PointNet/model/modelnet40-dense-gcn-3%%25noise-smooth-label-weighted-instance-rot-10deg-10%%25rescale-ensemble20/2020-12-11-15-20-26-4-346-0/opt-latest.save', \
    30
model_milestone, optim_milestone, beg_epochs = None, None, 0 # comment this if need to load from milestone

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def load_model(f: str, optim:str, e: int):
    global beg_epochs, model
    model.load_state_dict(torch.load(f))
    beg_epochs = e
    print("Loading milestone with epoch %d at %s" % (beg_epochs, f))
    if optim is not None:
        optimizer.load_state_dict(torch.load(optim))
        print("Loading milestone optimizer with epoch %d at %s" % (beg_epochs, optim))
    evaluate(model, msg="Loaded Model:Accuracy: [%.2f %%, %.2f %%] on [Train/Test] Dataset")
    
def parallel_cuda(batchs):
    batchs = [data.to(device) for data in batchs]
    labels = [data.y for data in batchs] # actually batchs
    # print(len(labels)) # assumably GPU count
    labels = torch.cat(labels).to(device)
    bs = labels.shape[0] 
    return batchs, labels, bs

def p_augment(parallel=True, *args):
    """
    Proxy for augments
    """
    if parallel:
        parallel_augment(*args)
    else:
        augment(*args)

def augment(batch, out, bs):
    global random_rotation, instance_rotation, random_pertubation, instance_scale
    with torch.no_grad():
        if random_rotation: 
            out.pos, _, _ = apply_random_rotation(batch.pos, rot_axis=random.choice([1, 2, 3]), std=1/36)
            # rotate along a randomly selected axis
        if instance_rotation:
            out.pos, _ = apply_instance_scatter_rotation(batch.pos, batch_size=bs, std=1/36)
        if random_pertubation:
            out.pos, _ = apply_random_pertubation(batch.pos, std=0.003)
            # pertubate point cloud
        if instance_scale:
            out.pos, _ = apply_random_scale(batch.pos, batch_size=bs, std=0.1)
        if random_flip:
            out.pos, _ = apply_scatter_random_flip(batch.pos, batch_size=bs)

def parallel_augment(batchs, out_batchs, bs):
    global random_rotation, instance_rotation, random_pertubation, instance_scale
    with torch.no_grad():
        for batch, out in zip(batchs, out_batchs):
            if random_rotation: 
                out.pos, _, _ = apply_random_rotation(batch.pos, rot_axis=random.choice([1, 2, 3]), std=1/36)
                # rotate along a randomly selected axis, real instance-wise
            if instance_rotation:
                out.pos, _ = apply_instance_scatter_rotation(batch.pos, batch_size=1, std=1/36)
            if random_pertubation:
                out.pos, _ = apply_random_pertubation(batch.pos, std=0.003)
                # pertubate point cloud
            if instance_scale:
                out.pos, _ = apply_random_scale(batch.pos, batch_size=1, std=0.1)
            if random_flip:
                out.pos, _ = apply_random_flip(batch.pos, flip_axis=random.choice([1, 2, 3]))

def rescale(batchs, out_batchs, scale=1.0, parallel=True):
    with torch.no_grad():
        if parallel:
            for batch, out in zip(batchs, out_batchs):
                out.pos = batch.pos * scale
        else:
            out_batchs.pos = batchs.pos * scale

def train(model, epoch: int):
    model.train()
    correct, total = 0, 0
    total_loss = 0
    for i, batch in enumerate(loader, 0):
        if parallel:
            batch, labels, bs = parallel_cuda(batch)
            parallel_augment(batch, batch, bs)
        else:
            batch = batch.to(device)
            labels, bs = batch.y, batch.y.shape[0]
            augment(batch, batch, bs)
        
        model.zero_grad()
        out = model(batch)
        if use_smooth_loss is True:
            loss = cal_loss(out, labels, label_weights=label_weights) # use smooth label weight, as in DGCNN
        elif use_unbal_lw is True:
            loss = F.nll_loss(out, labels, weight=label_weights)
        else:
            loss = F.nll_loss(out, labels)
        total_loss += loss.detach().item()
        correct += out.max(dim=1)[1].eq(labels).sum().item() # count correct samples (!here in train mode!)
        total += bs
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("[%d/%d]Loss: %.3f" % (epoch, i, loss.item()))
    scheduler.step()
    return total_loss / len(loader), correct / total


def evaluate(model, msg=None, eval_train=True):
    model.eval()

    with torch.no_grad():
        if eval_train:
            # validation, dataset too large, have to enumerate by batch
            print(colorama.Fore.BLUE + "Validating on train set...")
            correct, samples = 0, 0
            for i, batch in enumerate(loader, 0):
                if parallel:
                    labels, bs = parallel_cuda(batch)
                    parallel_augment(batch, batch, bs)
                else:
                    batch = batch.to(device)
                    labels, bs = batch.y, batch.y.shape[0]
                    augment(batch, batch, bs)
                out = model(batch)
                # print(out, out.device)
                correct += out.max(dim=1)[1].eq(labels).sum().item()  # count correct samples (in test set)
                samples += bs
            v_accu = correct / samples
            # print(correct, samples)
        else:
            v_accu = -1
        correct = 0
        print(colorama.Fore.BLUE + "Validating on test set...")
        for i, batch in enumerate(tqdm(test_loader), 0):
            if parallel:
                # batch = batch.cuda()
                # labels, bs = batch.y, batch.y.shape[0]
                labels, bs = parallel_cuda(batch)
            else:
                batch = batch.to(device)
                labels, bs = batch.y, batch.y.shape[0]
            if use_ensemble_test:
                out = ensemble_test(model, batch, bs)
            else:
                out = model(batch)
            correct += out.max(dim=1)[1].eq(labels).sum().item()  # count correct samples (in test set)
        t_accu = correct / len(test_dataset)
    if msg is not None:
        assert isinstance(msg, str), 'eval msg must be a string'
        print(msg % (100 * v_accu, 100 * t_accu))
    return v_accu, t_accu

def ensemble_test(model, batch, bs, n=5):
    with torch.no_grad():
        # labels, bs = batch.y, batch.y.shape[0]
        model.eval()
        if parallel:
            orig_batch = [data.clone() for data in batch] # save for future transformation
        else:
            orig_batch = batch.clone()
        out = model(batch) # the original one
        if not rescale_ensemble:
            for _ in range(n):
                p_augment(parallel=parallel)
                out += model(batch)
        else:
            for scale in np.linspace(0.75, 1.25, num=n):
                rescale(orig_batch, batch, scale=scale, parallel=parallel)
                out += model(batch)
        out /= n + 1
    return out

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # gpu_id = 6
    device = torch.device('cuda:%d' % (gpu_id) if torch.cuda.is_available() else 'cpu')
    print(colorama.Fore.BLUE + colorama.Back.YELLOW + "Training Controller: " + repr(device))
    
    # model_type = 'gcn'
    assert model_type in ('sgc', 'gcn', 'dense-gcn')
    # dataset choice
    assert dataset_type in ['10', '40']
    if dataset_type == '10':
        pl_path = 'modelnet-10-pointcloud'
        model_path = 'modelnet10-gcn'
        fout = 10
    elif dataset_type == '40':
        pl_path = 'modelnet40-knn'
        model_path = 'modelnet40-%s-noise3-flip-rot10-scale10-smooth-w-label-ens5RS-focal' % (model_type)
        fout = 40
    assert pl_path and model_path and fout
    timestamp = '%d-%d-%d-%d-%d-%d-%d-%d-%d' % time.localtime(time.time())
    model_path = os.path.join('/data/pkurei/PointNet/model/', model_path, timestamp)
    data_path = os.path.join('/data1/', pl_path)
    for path in (data_path, model_path):
        check_dir(path, color=colorama.Fore.CYAN)
    
    model = None
    if model_type == 'gcn':
        batch_size = 32
        model = GCNClassifierSparse(in_channels=3, classes=40) # train a 5-layer GCN => 65-70
    elif model_type == 'sgc':
        model = SGCClassifier(in_channels=3, classes=40, K=5) # train a 5-hops SGC => 80.2
        batch_size = 64
    elif model_type == 'dense-gcn':
        batch_size = 32 * ngpu
        model = DenseGCNClassifier(in_channels=3, classes=40) # train a 4-layer DenseGCN => 86.2 +- 0.2(unaugmented)
    
    if parallel:
        model = DataParallel(model, device_ids=gpu_ids, output_device=gpu_id)
        if use_sbn:
            try:
                from sync_batchnorm import convert_model
                model = convert_model(model)
                # fix sync-batchnorm
            except ModuleNotFoundError:
                raise ModuleNotFoundError("Sync-BN plugin not found")
        model = model.to(device)
    else:   
        model = model.to(device)
        
    train_dataset = ModelNet(root=data_path, name='40', train=True,
                            # pre_transform=tg.transforms.SamplePoints(samplePoints),
                            # transform=tg.transforms.KNNGraph(k=10))
                            pre_transform=transform(samplePoints=samplePoints, k=20))
    test_dataset = ModelNet(root=data_path, name='40', train=False,
                            # pre_transform=tg.transforms.SamplePoints(samplePoints),
                            # transform=tg.transforms.KNNGraph(k=10))
                            pre_transform=transform(samplePoints=samplePoints, k=20))
    if data_load_only:
        raise Exception("Dataset Loaded from %s" % data_path) # Stop here if only needs to
    else:
        print(colorama.Fore.CYAN + "Dataset Loaded from %s" % data_path)
    if parallel: 
        loader = DataListLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=16, pin_memory=True)
        test_loader = DataListLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=16, pin_memory=True)
    else:
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=16, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=16, pin_memory=True)

    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': 0.002}],
                                lr=0.002, weight_decay=5e-4, betas=(0.9, 0.999))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.65, last_epoch=beg_epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, last_epoch=beg_epochs)

    # load network milestone or initialize model weights
    if model_milestone is not None:
        load_model(model_milestone, optim_milestone, beg_epochs)
    else:
        init_weights(model)

    try:
        label_weights = torch.load(os.path.join(data_path, 'lw.pt'))  # literally data/DATASETNAME/lw.pt
        # raise Exception() # force reload each time
    except:
        obool = torch.backends.cudnn.deterministic # save state
        torch.backends.cudnn.deterministic = True
        train_y = train_dataset.data.y
        label_weights = train_y.bincount().float() / len(train_y)  # weights of each class
        print(train_y, label_weights, label_weights.shape)
        label_weights = label_weights.max() / label_weights  # w_c proportional to 1/|N_c|
        label_weights = label_weights / label_weights.sum() * len(label_weights)
        label_weights = label_weights.to(device)
        torch.save(label_weights, os.path.join(data_path, 'lw.pt'))
        torch.backends.cudnn.deterministic = obool # restore deterministics
    
    label_weights = label_weights.to(device)
    print(colorama.Fore.MAGENTA + "Label weights:")
    print(label_weights)

    timestamp = '%d-%d-%d-%d-%d-%d-%d-%d-%d' % time.localtime(time.time())
    writer = SummaryWriter(comment='%s-modelnet40' % model_type)  # global steps => index of epoch

    batch_cnt = len(loader)
    print(colorama.Fore.MAGENTA + "Begin training with: batch size %d, %d batches in total" % (batch_size, batch_cnt))

    # for i in trange(4):
    #     v_accu, t_accu = evaluate()
    #     print("[%d]Average Loss: %.3f, Accuracy: [%.2f %%, %.2f %%] on [Train/Test] Dataset"
    #           % (100, 100, 100 * v_accu, 100 * t_accu))


    for e in trange(beg_epochs, epochs):
        avg_loss, v_accu = train(model, e)
        gc.collect()
        _, t_accu = evaluate(model, eval_train=False)
        gc.collect()
        print(colorama.Fore.MAGENTA + "[%d]Average Loss: %.3f, Accuracy: [%.2f %%, %.2f %%] on [Train/Test] Dataset"
              % (e, avg_loss, 100 * v_accu, 100 * t_accu))

        # save model for each <milestone_period> epochs (i.e. 5 default)
        if e % milestone_period == 0 and e != 0:
            torch.save(model.state_dict(), os.path.join(model_path, 'model-%d.save' % (e)))
            torch.save(optimizer.state_dict(), os.path.join(model_path, 'opt-%d.save' % (e)))
            torch.save(model.state_dict(), os.path.join(model_path, 'model-latest.save'))
            torch.save(optimizer.state_dict(), os.path.join(model_path, 'opt-latest.save'))

        # log to tensorboard
        record_dict = {
            'loss': avg_loss,
            # 'accuracy': {
            'train_accuracy': v_accu,
            'test_accuracy': t_accu
            # }
        }

        for key in record_dict:
            if not isinstance(record_dict[key], dict):
                writer.add_scalar(key, record_dict[key], e)
            else: 
                writer.add_scalars(key, record_dict[key], e) # add multiple records
            
