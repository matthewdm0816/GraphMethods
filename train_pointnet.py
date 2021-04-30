from pointnet import *
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torch_geometric as tg
from torch_geometric.data import Data
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader
import os, sys, time
# from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from tqdm import *

def save_model(f: str):     # not used
    print("Saving model to file %s" % (f))
    torch.save(model, f)

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
    evaluate(msg="Loaded Model:Accuracy: [%.2f %%, %.2f %%] on [Train/Test] Dataset")

def train(epoch: int):
    model.train()
    scheduler.step()
    total_loss = 0
    for i, batch in enumerate(loader, 0):
        pcs, labels = batch
        pcs, labels = pcs.to(device), labels.to(device)
        model.zero_grad()
        out, t_in, t_feat = model(pcs)
        loss = F.nll_loss(out, labels, weight=label_weights) + transformation_feature_regularizer(t_feat) * 1e-3
        total_loss += loss.detach().item() * pcs.shape[0]
        # correct += out.max(dim=1)[1].eq(labels).sum() # count correct samples (!here in train mode!)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("[%d/%d]Loss: %.3f" % (epoch, i, loss.item()))
    return total_loss / len(train_dataset)
    #  correct / len(loader)

def evaluate(msg=None):
    model.eval()
    with torch.no_grad():
        # validation, dataset too large, have to enumerate by batch
        correct, samples = 0, 0
        for i, batch in enumerate(loader, 0):
            pcs, labels = batch
            pcs, labels = pcs.to(device), labels.to(device)
            out, t_in, t_feat = model(pcs)
            correct += out.max(dim=1)[1].eq(labels).sum().item()  # count correct samples (in test set)
            samples += pcs.shape[0]
        v_accu = correct / samples
        # print(correct, samples)
        correct = 0
        for i, batch in enumerate(test_loader, 0):
            pcs, labels = batch
            pcs, labels = pcs.to(device), labels.to(device)
            out, t_in, t_feat = model(pcs)
            correct += out.max(dim=1)[1].eq(labels).sum().item()  # count correct samples (in test set)
        t_accu = correct / len(test_dataset)
    if msg is not None:
        assert isinstance(msg, str), 'eval msg must be a string'
        print(msg % (100 * v_accu, 100 * t_accu))
    return v_accu, t_accu


batch_size = 60
epochs = 2001
show_params = False
milestone_period = 5

dataset_type = '40'
assert dataset_type in ['10', '40']
if dataset_type == '10':
    pl_path = '/data/pkurei/PointNet/modelnet-10-2500/pointcloud'
    model_path = 'modelnet10-pointnet'
    fout = 10
elif dataset_type == '40':
    pl_path = '/data/pkurei/PointNet/modelnet40-2500/pointcloud'
    model_path = 'modelnet40-pointnet'
    fout = 40
assert pl_path and model_path and fout
mp = os.path.join('/data/pkurei/PointNet/model', model_path)
if not os.path.exists(mp):
    os.makedirs(mp)

model_milestone, optim_milestone, beg_epochs = 'model/modelnet40-pointnet/2020-9-24-20-52-31-3-268-0.mdl125', \
                                               'model/modelnet40-pointnet/2020-9-24-20-52-31-3-268-0.opt125', \
                                               125
model_milestone, optim_milestone, beg_epochs = None, None, 0 # comment this if need to load from milestone


if __name__ == '__main__':
    if not os.path.exists('model'):
        os.mkdir('model')

    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

    model = PointNet(fout=fout).to(device)
    # no regularization here, use reg explicitly
    # Adam Ooptimizer
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': 0.002}],
                           lr=0.002, weight_decay=0, betas=(0.9, 0.999))
    # optimizer = optim.SGD([{'params': model.parameters(), 'initial_lr': 0.01}],
    #                       lr=0.01, nesterov=True, weight_decay=0, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.65, last_epoch=beg_epochs)

    # load dataset
    # train_dataset = ModelNet(root='data/modelnet40-1024', name='40', train=True,
    #                          # pre_transform=tg.transforms.SamplePoints(samplePoints),
    #                          # transform=tg.transforms.KNNGraph(k=10))
    #                          pre_transform=tg.transforms.SamplePoints(1024))
    # test_dataset = ModelNet(root='data/modelnet40-1024', name='40', train=False,
    #                         # pre_transform=tg.transforms.SamplePoints(samplePoints),
    #                         # transform=tg.transforms.KNNGraph(k=10))
    #                         pre_transform=tg.transforms.SamplePoints(1024))
    train_tensor, test_tensor = \
        torch.load(os.path.join(pl_path, 'train.points')), \
        torch.load(os.path.join(pl_path, 'test.points'))
    # print(train_tensor.shape)
    train_y, test_y = \
        torch.load(os.path.join(pl_path, 'train.labels')).long(), \
        torch.load(os.path.join(pl_path, 'test.labels')).long()
    train_tensor, test_tensor = train_tensor.transpose(2, 1), test_tensor.transpose(2, 1)
    label_weights = train_y.bincount().float() / len(train_y) # weights of each class
    label_weights = label_weights.max() / label_weights  # w_c proportional to 1/|N_c|
    label_weights = label_weights / label_weights.sum() * len(label_weights)
    label_weights = label_weights.to(device)
    torch.save(label_weights, os.path.join(pl_path, 'lw.pt'))
    train_dataset = ModelNet10(train_tensor, train_y)
    test_dataset = ModelNet10(test_tensor, test_y)

    loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # load network milestone or initialize model weights
    if model_milestone is not None:
        load_model(model_milestone, optim_milestone ,beg_epochs)
    else:
        init_weights(model)

    timestamp = '%d-%d-%d-%d-%d-%d-%d-%d-%d' % time.localtime(time.time())
    writer = SummaryWriter()  # global steps => index of epoch

    if show_params:
        summary(model, input_size=(3, 2500))

    batch_cnt = len(loader)

    print("Begin training with: batch size %d, %d batches in total" % (batch_size, batch_cnt))
    v_accu, t_accu = evaluate()
    print("\n[%d]Accuracy: [%.2f %%, %.2f %%] on [Train/Test] Dataset"
          % (-1, 100 * v_accu, 100 * t_accu))
    for e in trange(beg_epochs, epochs):
        # train model
        avg_loss = train(e)
        v_accu, t_accu = evaluate()
        print("\n[%d]Average Loss: %.3f, Accuracy: [%.2f %%, %.2f %%] on [Train/Test] Dataset"
              % (e, avg_loss, 100 * v_accu, 100 * t_accu))

        # save model for each <milestone_period> epochs (i.e. 5 default)
        if e % milestone_period == 0 and e != 0:
            torch.save(model.state_dict(), os.path.join(mp, '%s.mdl%d' % (timestamp, e)))
            torch.save(optimizer.state_dict(), os.path.join(mp, '%s.opt%d' % (timestamp, e)))

        # log to tensorboard
        record_dict = {
            'loss': avg_loss,
            'test_accuracy': v_accu,
            'train_accuracy': t_accu
        }

        for key in record_dict:
            writer.add_scalar(key, record_dict[key], e)


