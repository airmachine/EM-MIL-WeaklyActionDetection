# --------------------------------------------------------
# EM-MIL
# Copyright (c) 2021 University of California, Berkeley
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhekun Luo
# --------------------------------------------------------

import os
import torch
from model import generate_model
from train import train_epoch
from opts import parse_opts
from weaklyDataset import weaklyDataset
from torch.utils.data import Dataset, DataLoader


def main():
    print('')
    print("training EM model")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    opt = parse_opts()
    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    optimizer = torch.optim.Adam(parameters, lr=opt.learning_rate)

    if not os.path.exists(opt.model_weight):
        os.mkdir(opt.model_weight)

    trainSet = weaklyDataset(opt.train_path)
    train_loader = DataLoader(trainSet, batch_size=1,
                              shuffle=True, num_workers=0)
    E_step = False

    def adjust_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.learning_rate*4

    for epoch in range(1, 66):

        if epoch <= 10:
            stage = 1
            E_step = False

        elif epoch > 10 and epoch <= 20:
            stage = 1
            E_step = True

        elif epoch > 20 and epoch <= 30:
            stage = 2
            E_step = False
            adjust_learning_rate(optimizer)

        else:
            stage = 2
            E_step = not E_step

        train_epoch(epoch, train_loader, model, optimizer, opt, E_step, stage)
        torch.save(model.state_dict(), opt.model_weight+"/{}.pt".format(epoch))
    return
