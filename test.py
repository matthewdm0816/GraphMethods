from train_pointnet import *
import math, os, sys, random

if __name__ == '__main__':
    model = PointNet(fout=10).to(device)

    # load dataset
    train_tensor, test_tensor = torch.load(pl_path + 'train.points'), \
                                torch.load(pl_path + 'test.points')
    train_y, test_y = torch.load(pl_path + 'train.labels').long(), \
                      torch.load(pl_path + 'test.labels').long()
    train_tensor, test_tensor = train_tensor.transpose(2, 1), test_tensor.transpose(2, 1)
    train_dataset = ModelNet10(train_tensor, train_y)
    test_dataset = ModelNet10(test_tensor, test_y)
    loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # load network milestone
    load_model(model_milestone, beg_epochs)
    # UNIMPLEMENTED... TODO