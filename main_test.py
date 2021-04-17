# --------------------------------------------------------
# EM-MIL
# Copyright (c) 2021 University of California, Berkeley
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhekun Luo
# --------------------------------------------------------

import os
import torch
from model import generate_model
from model_unt import generate_model_unt
from test import test
from opts import parse_opts
from weaklyDataset import weaklyDataset
from torch.utils.data import Dataset, DataLoader


def main_test():
    print('')
    print('start testing')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    opt = parse_opts()

    model, parameters = generate_model(opt)
    model_unt, parameters_unt = generate_model_unt(opt)

    model.load_state_dict(torch.load(opt.model_weight+"/65.pt"))
    model.eval()

    model_unt.load_state_dict(torch.load(opt.model_unt_weight+"/45.pt"))
    model_unt.eval()

    testSet = weaklyDataset(opt.test_path)
    test_loader = DataLoader(testSet, batch_size=1,
                             shuffle=False, num_workers=0)
    test(test_loader, model, model_unt, opt)
    return
