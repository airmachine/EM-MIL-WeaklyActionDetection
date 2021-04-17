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

# this script contains the UntrimmedNet model for action prediction
# UntrimmedNet paper: https://arxiv.org/abs/1703.03329
# Here we re-implement its idea and use a class-specific attention for UntrimmedNet

# citation of UntrimmedNet:
# @article{DBLP:journals/corr/WangXLG17,
#   author    = {Limin Wang and
#                Yuanjun Xiong and
#                Dahua Lin and
#                Luc Van Gool},
#   title     = {UntrimmedNets for Weakly Supervised Action Recognition and Detection},
#   journal   = {CoRR},
#   volume    = {abs/1703.03329},
#   year      = {2017},
#   url       = {http://arxiv.org/abs/1703.03329},
#   archivePrefix = {arXiv},
#   eprint    = {1703.03329},
#   timestamp = {Mon, 16 Mar 2020 17:55:51 +0100},
#   biburl    = {https://dblp.org/rec/journals/corr/WangXLG17.bib},
#   bibsource = {dblp computer science bibliography, https://dblp.org}
# }


def generate_model_unt(opt):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_unt = UntrimmedNet(device=device, opt=opt).to(device)
    torch.backends.cudnn.enabled = False
    return model_unt, model_unt.parameters()


class UntrimmedNet(nn.Module):

    def __init__(self, device, opt):
        super(UntrimmedNet, self).__init__()

        self.device = device
        self.opt = opt
        self.K = self.opt.n_classes
        self.D = self.opt.dim
        self.blend = 0.5

        self.att_rgb = nn.Sequential(
            nn.Linear(self.D, 1), nn.Softmax(dim=0)).to(self.device)
        self.att_flow = nn.Sequential(
            nn.Linear(self.D, 1), nn.Softmax(dim=0)).to(self.device)

        self.cls_rgb = nn.Sequential(
            nn.Linear(self.D, self.K), nn.Softmax(dim=1)).to(self.device)
        self.cls_flow = nn.Sequential(
            nn.Linear(self.D, self.K), nn.Softmax(dim=1)).to(self.device)

    def forward(self, rgb, flow):

        a_rgb = self.att_rgb(rgb)
        a_flow = self.att_flow(flow)

        c_rgb = self.cls_rgb(rgb)
        c_flow = self.cls_flow(flow)

        a = self.blend*a_rgb + (1-self.blend)*a_flow
        c = self.blend*c_rgb + (1-self.blend)*c_flow

        cas = a*c
        # cas = (T,K)
        pred = torch.sum(cas, dim=0)
        # pred = (K)
        return pred