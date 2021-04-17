# --------------------------------------------------------
# EM-MIL
# Copyright (c) 2021 University of California, Berkeley
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhekun Luo
# --------------------------------------------------------

import os
import torch
import torch.nn.functional as F


def test(data_loader, model, model_unt, opt):

    model.eval()
    model_unt.eval()
    convert_ratio = float(opt.downsample_rate * opt.frame_per_clip / opt.fps)
    device = model.device

    if os.path.exists(opt.result_path):
        os.remove(opt.result_path)

    with torch.no_grad():
        with open(opt.result_path, 'w') as outfile:
            for i, (rgb, flow, _, name) in enumerate(data_loader):

                rgb = rgb[0].to(device)
                flow = flow[0].to(device)
                name = str(name[0])

                if rgb.shape[0] != flow.shape[0]:
                    cut = min(rgb.shape[0], flow.shape[0])
                    rgb = rgb[:cut, :]
                    flow = flow[:cut, :]

                k_rgb, k_flow, c_rgb, c_flow = model.forward_body(rgb, flow)
                key = (k_rgb + k_flow)/2
                cls = (c_rgb + c_flow)/2
                T, K = cls.shape

                pred = model_unt(rgb, flow)

                for i in range(K):
                    if pred[i] < 0.15:
                        continue

                    proposal_open = False
                    start = 0
                    end = 0

                    k_i = key[:, 0]
                    c_i = cls[:, i]

                    opt._lambda = 0.8
                    score = opt._lambda*k_i + opt._lambda*c_i

                    size = torch.max(score) - torch.min(score)
                    score = (score - torch.mean(score)) / size

                    for j in range(T):
                        activation = float(score[j])
                        if activation > 0.05:
                            if not proposal_open:
                                proposal_open = True
                                start = j

                        else:
                            if proposal_open:
                                end = j

                                duration = end - start + 1
                                start = float(start) * convert_ratio
                                end = float(end) * convert_ratio
                                vid = name

                                outfile.write("{} {} {} {} {}".format(
                                    vid, round(start, 2), round(end, 2), i+1, 1))
                                outfile.write('\n')

                                # reset
                                proposal_open = False
                                start = 0
                                end = 0
