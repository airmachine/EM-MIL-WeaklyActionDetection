# --------------------------------------------------------
# EM-MIL
# Copyright (c) 2021 University of California, Berkeley
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhekun Luo
# --------------------------------------------------------

from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class weaklyDataset(Dataset):

    def __init__(self, root, transform=None):

        self.root = root

        self.rgb_root = self.root + "/" + "X_RGB"
        self.flow_root = self.root + "/" + "X_flow"
        self.label_root = self.root + "/" + "Y_flow"
        self.name_root = self.root + "/" + "Name_flow"

        self.x_list = os.listdir(self.rgb_root)
        self.length = len(self.x_list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb_path = self.rgb_root + "/" + str(idx) + ".pt"
        flow_path = self.flow_root + "/" + str(idx) + ".pt"
        label_path = self.label_root + "/" + str(idx) + ".pt"
        name_path = self.name_root + "/" + str(idx) + ".pt"

        rgb = torch.load(rgb_path)
        flow = torch.load(flow_path)
        label = torch.load(label_path)
        name = torch.load(name_path)

        return rgb, flow, label, name

