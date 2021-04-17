# --------------------------------------------------------
# EM-MIL
# Copyright (c) 2021 University of California, Berkeley
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhekun Luo
# --------------------------------------------------------

import os
import torch
from model_unt import generate_model_unt
from train_unt import train_unt_epoch
from opts import parse_opts
from weaklyDataset import weaklyDataset
from torch.utils.data import Dataset, DataLoader
from check_untrimmed import check_untrimmed


def main_unt():
    print('')
    print("training UntrimmedNet model")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    opt = parse_opts()

    model_unt, parameters_unt = generate_model_unt(opt)
    optimizer_unt = torch.optim.Adam(parameters_unt, lr=opt.learning_rate)

    if not os.path.exists(opt.model_unt_weight):
        os.mkdir(opt.model_unt_weight)

    trainSet = weaklyDataset(opt.train_path)
    train_loader = DataLoader(trainSet, batch_size=1,
                              shuffle=True, num_workers=0)

    for epoch in range(1, 46):
        r = train_unt_epoch(epoch, train_loader, model_unt, optimizer_unt, opt)
        torch.save(model_unt.state_dict(),
                   opt.model_unt_weight+"/{}.pt".format(epoch))