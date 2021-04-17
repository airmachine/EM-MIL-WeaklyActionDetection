# --------------------------------------------------------
# EM-MIL
# Copyright (c) 2021 University of California, Berkeley
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhekun Luo
# --------------------------------------------------------

from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

# this script contains the source code for EM-MIL model


def generate_model(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EM_MIL(device=device, opt=opt).to(device)
    torch.backends.cudnn.enabled = False
    return model, model.parameters()


class EM_MIL(nn.Module):

    def __init__(self, device, opt):

        super(EM_MIL, self).__init__()
        self.device = device
        self.opt = opt
        self.K = self.opt.n_classes
        self.D = self.opt.dim
        self.BCE = torch.nn.BCELoss()
        self._lambda = opt._lambda
        self._gamma = opt._gamma

        # key instance assignment
        self.key_rgb = nn.Sequential(
            nn.Conv1d(self.D, 256, kernel_size=3, stride=1, padding=1, bias=True))
        self.key_flow = nn.Sequential(
            nn.Conv1d(self.D, 256, kernel_size=3, stride=1, padding=1, bias=True))
        self.key_tail_rgb = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.key_tail_flow = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # classification
        self.cls_rgb = nn.Sequential(
            nn.Conv1d(self.D, 256, kernel_size=3, stride=1, padding=1, bias=True))
        self.cls_flow = nn.Sequential(
            nn.Conv1d(self.D, 256, kernel_size=3, stride=1, padding=1, bias=True))
        self.cls_tail_rgb = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, self.K),
            nn.Sigmoid()
        )
        self.cls_tail_flow = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, self.K),
            nn.Sigmoid()
        )

    def forward_body(self, rgb, flow):

        T = rgb.shape[0]

        # rgb/flow = (T, D)
        rgb = rgb.permute(1, 0).expand(1, self.D, T)
        flow = flow.permute(1, 0).expand(1, self.D, T)
        # rgb/flow = (T, D) -> (D, T) -> (1, D, T)

        # key instance assignment for rgb
        k_rgb = self.key_rgb(rgb).view(-1, T).permute(1, 0)
        # k_rgb = (1, D, T) -> (1, 256, T) -> (256, T) -> (T, 256)
        k_rgb = self.key_tail_rgb(k_rgb)
        # k_rgb = (T, 1)

        # key instance assignment for flow
        k_flow = self.key_flow(flow).view(-1, T).permute(1, 0)
        # k_flow = (1, D, T) -> (1, 256, T) -> (256, T) -> (T, 256)
        k_flow = self.key_tail_flow(k_flow)
        # k_flow = (T, 1)

        # classification for rgb
        c_rgb = self.cls_rgb(rgb).view(-1, T).permute(1, 0)
        # c_rgb = (1, D, T) -> (1, 256, T) -> (256, T) -> (T, 256)
        c_rgb = self.cls_tail_rgb(c_rgb)
        # c_rgb = (T, K)

        # classification for flow
        c_flow = self.cls_flow(flow).view(-1, T).permute(1, 0)
        # c_flow = (1, D, T) -> (1, 256, T) -> (256, T) -> (T, 256)
        c_flow = self.cls_tail_flow(c_flow)
        # c_flow = (T, K)

        return k_rgb, k_flow, c_rgb, c_flow

    def forward_M(self, rgb, flow, label, stage):

        T = rgb.shape[0]
        k_rgb, k_flow, c_rgb, c_flow = self.forward_body(rgb, flow)
        loss = 0

        if stage == 1:
            zero_label = torch.zeros(T).to(self.device)
            one_label = torch.ones(T).to(self.device)

        elif stage == 2:
            zero_label = torch.zeros(T).to(self.device)

            key = (k_rgb[:, 0] + k_flow[:, 0])/2
            key = key.clone().detach()
            threshold = torch.mean(key) + self._gamma * \
                (torch.max(key) - torch.min(key))
            one_label = torch.zeros(T).to(self.device)
            one_label[key > threshold] = 1

        for i in range(self.K):
            # rgb component
            c_rgb_i = c_rgb[:, i] # c_rgb_i = (T)
            if label[i] == 1:
                loss += self.BCE(c_rgb_i, one_label)
            else:
                loss += self.BCE(c_rgb_i, zero_label)

            # flow component
            c_flow_i = c_flow[:, i]
            if label[i] == 1:
                loss += self.BCE(c_flow_i, one_label)
            else:
                loss += self.BCE(c_flow_i, zero_label)

        key = (k_rgb + k_flow)/2
        cls = (c_rgb + c_flow)/2

        return loss, key, cls

    def forward_E(self, rgb, flow, label, step):

        T = rgb.shape[0]
        k_rgb, k_flow, c_rgb, c_flow = self.forward_body(rgb, flow)
        cls = (c_rgb + c_flow)/2
        pseudo_label = torch.zeros(T).to(self.device)

        for i in range(self.K):
            if label[i] == 1:
                cls_i = cls[:, i] # cls_i = (K)
                threshold = torch.mean(cls_i)
                pseudo_i = (cls_i > threshold).clone().detach().type(
                    'torch.FloatTensor').to(self.device)
                pseudo_label = torch.max(pseudo_label, pseudo_i)

        loss = 0
        loss += self.BCE(k_rgb.view(-1), pseudo_label)
        loss += self.BCE(k_flow.view(-1), pseudo_label)

        key = (k_rgb + k_flow)/2
        cls = (c_rgb + c_flow)/2

        return loss, key, cls