from dgcnn import *
from helpers import *
from tqdm import *
from torch.utils.tensorboard import SummaryWriter
import os, sys, time, random
import torch.optim as optim
from train_pointnet import save_model, load_model, init_weights
from dataloader import scale_normalize

# parameters and hyper-parameters
dataset_type = '40'
samplePoints = 1024
batch_size = 48  # largest affordable on GTX1080 with 8G VRAM quq, even with sparse
epochs = 501
milestone_period = 5

data_load_only = False

# model_milestone, optim_milestone, beg_epochs = 'model/pointnet-2020-8-12-23-7-53-2-225-0.mdl', None, 210
model_milestone, optim_milestone, beg_epochs = None, None, 0 # comment this if need to load from milestone

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(epoch: int):
    model.train()
    total_loss = 0
    for i, batch in enumerate(loader, 0):
        with torch.no_grad():
            batch = batch.to(device)
        labels, bs = batch.y, batch.y.shape[0]
        model.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out, labels, weight=label_weights)
        total_loss += loss.detach().item()
        # correct += out.max(dim=1)[1].eq(labels).sum() # count correct samples (!here in train mode!)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("[%d/%d]Loss: %.3f" % (epoch, i, loss.item()))
    scheduler.step()
    return total_loss / len(loader)


def evaluate(msg=None, eval_train=True):
    model.eval()
    with torch.no_grad():
        if eval_train:
            # validation, dataset too large, have to enumerate by batch
            correct, samples = 0, 0
            for i, batch in enumerate(loader, 0):
                batch = batch.to(device)
                labels, bs = batch.y, batch.y.shape[0]
                out = model(batch)
                correct += out.max(dim=1)[1].eq(labels).sum().item()  # count correct samples (in test set)
                samples += bs
            v_accu = correct / samples
            # print(correct, samples)
        else:
            v_accu = -1
        correct = 0
        for i, batch in enumerate(test_loader, 0):
            with torch.no_grad():
                batch = batch.to(device)
            labels, bs = batch.y, batch.y.shape[0]
            out = model(batch)
            correct += out.max(dim=1)[1].eq(labels).sum().item()  # count correct samples (in test set)
        t_accu = correct / len(test_dataset)
    if msg is not None:
        assert isinstance(msg, str), 'eval msg must be a string'
        print(msg % (100 * v_accu, 100 * t_accu))
    return v_accu, t_accu


if __name__ == '__main__':

    # dataset choice
    assert dataset_type in ['10', '40']
    if dataset_type == '10':
        pl_path = 'modelnet-10-pointcloud'
        model_path = 'modelnet10-dgcnn'
        fout = 10
    elif dataset_type == '40':
        pl_path = 'modelnet40-1024-normal'
        model_path = 'modelnet40-dgcnn'
        fout = 40
    assert pl_path and model_path and fout
    model_path = os.path.join('/data/pkurei/PointNet/model/', model_path)
    data_path = os.path.join('/data/pkurei/PointNet/', pl_path)
    for path in (data_path, model_path):
        check_dir(path)

    model = DGCNNClassifier(in_channels=3, classes=40).to(device)
    train_dataset = ModelNet(root=data_path, name='40', train=True,
                            # pre_transform=tg.transforms.SamplePoints(samplePoints),
                            # transform=tg.transforms.KNNGraph(k=10))
                            pre_transform=scale_normalize)
    test_dataset = ModelNet(root=data_path, name='40', train=False,
                            # pre_transform=tg.transforms.SamplePoints(samplePoints),
                            # transform=tg.transforms.KNNGraph(k=10))
                            pre_transform=scale_normalize)

    if data_load_only:
        raise Exception("Dataset Loaded")  # Stop here if only needs to
    else:
        print("Dataset Loaded")
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': 0.002}],
                           lr=0.002, weight_decay=2e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.65, last_epoch=beg_epochs)

    # load network milestone or initialize model weights
    if model_milestone is not None:
        load_model(model_milestone, optim_milestone, beg_epochs)
    else:
        init_weights(model)

    try:
        label_weights = torch.load(os.path.join(data_path, 'lw.pt'))  # literally data/DATASETNAME/lw.pt
    except:
        train_y = train_dataset.data.y
        label_weights = train_y.bincount().float() / len(train_y)  # weights of each class
        label_weights = label_weights.max() / label_weights  # w_c proportional to 1/|N_c|
        label_weights = label_weights / label_weights.sum() * len(label_weights)
        label_weights = label_weights.to(device)
        torch.save(label_weights, os.path.join(data_path, 'lw.pt'))

    timestamp = '%d-%d-%d-%d-%d-%d-%d-%d-%d' % time.localtime(time.time())
    writer = SummaryWriter(comment='dgcnn-modelnet40')  # global steps => index of epoch

    batch_cnt = len(loader)
    print("Begin training with: batch size %d, %d batches in total" % (batch_size, batch_cnt))

    # for i in trange(4):
    #     v_accu, t_accu = evaluate()
    #     print("[%d]Average Loss: %.3f, Accuracy: [%.2f %%, %.2f %%] on [Train/Test] Dataset"
    #           % (100, 100, 100 * v_accu, 100 * t_accu))

    for e in trange(beg_epochs, epochs):
        avg_loss = train(e)
        v_accu, t_accu = evaluate(eval_train=True)
        print("[%d]Average Loss: %.3f, Accuracy: [%.2f %%, %.2f %%] on [Train/Test] Dataset"
              % (e, avg_loss, 100 * v_accu, 100 * t_accu))

        # save model for each <milestone_period> epochs (i.e. 5 default)
        if e % milestone_period == 0 and e != 0:
            torch.save(model.state_dict(), os.path.join(model_path, '%s.mdl%d' % (timestamp, e)))
            torch.save(optimizer.state_dict(), os.path.join(model_path, '%s.opt%d' % (timestamp, e)))

        # log to tensorboard
        record_dict = {
            'loss': avg_loss,
            'train_accuracy': v_accu,
            'test_accuracy': t_accu
        }

        for key in record_dict:
            writer.add_scalar(key, record_dict[key], e)