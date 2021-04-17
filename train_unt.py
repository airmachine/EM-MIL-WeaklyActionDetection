# --------------------------------------------------------
# EM-MIL
# Copyright (c) 2021 University of California, Berkeley
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhekun Luo
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


def train_unt_epoch(epoch, data_loader, model, optimizer, opt):

    print("epoch: {}".format(epoch))
    model.train()
    device = model.device
    batch_loss = 0
    bath_size = 1

    for i, (rgb, flow, label, name) in enumerate(data_loader):

        optimizer.zero_grad()
        rgb = rgb[0].to(device)
        flow = flow[0].to(device)
        label = label[0].to(device)
        name = str(name[0])
        # rgb/flow = (T,D)
        # label = (K)

        if rgb.shape[0] != flow.shape[0]:
            cut = min(rgb.shape[0], flow.shape[0])
            rgb = rgb[:cut, :]
            flow = flow[:cut, :]

        # pred = (K)
        pred = model(rgb, flow)
        # label = (K)
        label = F.normalize(label, p=1, dim=0)

        loss = torch.log(pred)*label
        loss = torch.sum(loss)

        batch_loss += loss

        if (i+1) % opt.batch_size == 0:
            batch_loss.backward()
            optimizer.step()
            batch_loss = 0
