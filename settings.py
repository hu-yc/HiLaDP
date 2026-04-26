#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import sys
import time
import logging
import numpy as np
import torch
import random
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


script_name = os.path.basename(sys.argv[0])

# base arguments
parser = argparse.ArgumentParser(description=f"Arguments for {script_name}")
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=123, metavar='S', help='random seed (default: 123)')
parser.add_argument('--data_path', type=str, default='../../dataset/', help='dataset path')
parser.add_argument('--log_dir', type=str, default='logs', help='Export path')
parser.add_argument('--ds', type=str, default='CIFAR10', help='aws, datarade, ...')
parser.add_argument('--bs', type=int, default=64, help='batch size for training')
parser.add_argument("--nr_fea", type=int, default=512, help='number of features')
parser.add_argument("--nr_class", type=int, default=10, help='number of classes')
parser.add_argument("--lr", type=float, default=2e-4, help='learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay [1e-4]')
parser.add_argument("--epochs", type=int, default=50, help='number of epochs to train')
parser.add_argument('--debug', type=str2bool, default=False, metavar='Debug Flag', help='debug flagm True or False')
# specific arguments
if script_name == 'prior_generator.py':
    parser.add_argument('--trunc', default=10, type=int, help='truncation for DPMM')
    parser.add_argument('--eta', default=1.0, type=float, help='eta for DPMM')
elif script_name == 'main.py':
    parser.add_argument('--eps', default=1.0, type=float, help='Privacy Budget')
    parser.add_argument('--eta', default=1.0, type=float, help='eta for DPMM')
    parser.add_argument('--w_cls', default=1.0, type=float, help='weight for KL divergence loss')
    parser.add_argument('--w_rec', default=1.0, type=float, help='weight for Rec divergence loss')
    parser.add_argument('--drop_ratio', default=0.1, type=float, help='dropout ratio value')

args = parser.parse_args()
out_dir = args.log_dir
if not args.no_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    args.cuda = True
else:
    device = torch.device("cpu")
    args.cuda = False
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Declare the log file
LOG_FORMAT = "[%(asctime)s %(filename)s:%(lineno)s]%(levelname)s >>> %(message)s"
if script_name == 'prior_generator.py':
    log_name = "prior_{}_{}_lr{}epoch{}trunc{}eta{}.log".format(
    args.ds, time.strftime("%Y%m%d_%H%M%S", time.localtime()),
    "".join(str(args.lr).split(".")), args.epochs, args.trunc, args.eta,
    )
else: 
    log_name = "{}_{}_lr{}epoch{}ncls{}eta{}eps{}drop{}.log".format(
    args.ds, time.strftime("%Y%m%d_%H%M%S", time.localtime()),
    "".join(str(args.lr).split(".")), args.epochs, args.nr_class, args.eta,
    args.eps, "".join(str(args.drop_ratio).split("."))
    )

if args.debug:
    logging.basicConfig(
        level=logging.DEBUG,
        format=LOG_FORMAT,
        stream=sys.stdout
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(out_dir, log_name),
        filemode="w",
        format=LOG_FORMAT
    )

logging.info("Record the params {}\n".format(args))

# Set the random seed
def seed_everything(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python hash seed
    # torch.use_deterministic_algorithms(True)  # Enable deterministic algorithms
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # Set CUBLAS workspace
