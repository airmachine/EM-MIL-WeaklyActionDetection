# --------------------------------------------------------
# EM-MIL
# Copyright (c) 2021 University of California, Berkeley
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhekun Luo
# --------------------------------------------------------

import argparse
# this script is for adding arguments
# also provides default param


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_path',
        default="validationSet",
        type=str,
        help='data path of train set data'
    )
    parser.add_argument(
        '--test_path',
        default="testSet",
        type=str,
        help='data path of test set data'
    )
    parser.add_argument(
        '--result_path',
        default='results.txt',
        type=str,
        help='dst to write result for eval'
    )
    parser.add_argument(
        '--n_classes',
        default=20,
        type=int,
        help='number of classes (thumos14: 20)'
    )
    parser.add_argument(
        '--dim',
        default=1024,
        type=int,
        help='dimension of input features loaded, for rgb/flow'
    )
    parser.add_argument(
        '--learning_rate',
        default=0.0001,
        type=float,
        help='default initial learning rate (later will change in main.py)'
    )
    parser.add_argument(
        '--manual_seed',
        default=1,
        type=int,
        help='random seed'
    )
    parser.add_argument(
        '--model_weight',
        default="model_weight",
        type=str,
        help='dir for saving the EM model weight'
    )
    parser.add_argument(
        '--model_unt_weight',
        default="model_unt_weight",
        type=str,
        help='dir for saving the UntrimmedNet model weight'
    )
    parser.add_argument(
        '--downsample_rate',
        default=2,
        type=int,
        help='sample one frame every * frame'
    )
    parser.add_argument(
        '--frame_per_clip',
        default=15,
        type=int,
        help='every clip contains * sampled frame'
    )
    parser.add_argument(
        '--fps',
        default=24.0,
        type=float,
        help='fps for the video'
    )
    parser.add_argument(
        '--_lambda',
        default=0.8,
        type=float,
        help='hyper parameter lambda, in Eq.9'
    )
    parser.add_argument(
        '--_gamma',
        default=0.15,
        type=float,
        help='hyper parameter gamma, in Eq.5'
    )
    parser.add_argument(
        '--batch_size',
        default=1,
        type=int,
        help='batch size for training'
    )

    args = parser.parse_args()
    return args