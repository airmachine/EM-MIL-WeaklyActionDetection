# --------------------------------------------------------
# EM-MIL
# Copyright (c) 2021 University of California, Berkeley
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhekun Luo
# --------------------------------------------------------

import torch


def train_epoch(epoch, data_loader, model, optimizer, opt, E_step, stage):
    print("#"*20)
    print("epoch: {}".format(epoch))

    model.train()
    device = model.device
    for param in model.parameters():
        param.requires_grad = False

    if E_step:
        print("E step")
        for param in model.key_rgb.parameters():
            param.requires_grad = True
        for param in model.key_flow.parameters():
            param.requires_grad = True
        for param in model.key_tail_rgb.parameters():
            param.requires_grad = True
        for param in model.key_tail_flow.parameters():
            param.requires_grad = True

    else:
        print("M step")
        for param in model.cls_rgb.parameters():
            param.requires_grad = True
        for param in model.cls_flow.parameters():
            param.requires_grad = True
        for param in model.cls_tail_rgb.parameters():
            param.requires_grad = True
        for param in model.cls_tail_flow.parameters():
            param.requires_grad = True

    batch_loss = 0
    for i, (rgb, flow, label, name) in enumerate(data_loader):

        optimizer.zero_grad()
        rgb = rgb[0].to(device)
        flow = flow[0].to(device)
        label = label[0].to(device)
        name = str(name[0])

        if rgb.shape[0] != flow.shape[0]:
            cut = min(rgb.shape[0], flow.shape[0])
            rgb = rgb[:cut, :]
            flow = flow[:cut, :]

        if (torch.sum(label) < 0.1):
            continue

        if E_step:
            loss, key, cls = model.forward_E(rgb, flow, label, stage)
        else:
            loss, key, cls = model.forward_M(rgb, flow, label, stage)

        batch_loss += loss

        if (i+1) % opt.batch_size == 0:
            batch_loss.backward()
            optimizer.step()
            batch_loss = 0
