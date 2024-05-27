import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import argparse
import json
import sys
import random
import numpy as np
import os
import time

def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        data = data.view(data.size(0), data.size(1), -1)
        channels_sum += data.mean(dim=[0, 2])
        channels_squared_sum += (data ** 2).mean(dim=[0, 2])
        num_batches += 1

    # 최종 평균
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Logger(object):
    def __init__(self, no_save=False):
        self.terminal = sys.stdout
        self.file = None
        self.no_save = no_save
    def open(self, fp, mode=None):
        if mode is None: mode = 'w'
        if not self.no_save: 
            self.file = open(fp, mode)
    def write(self, msg, is_terminal=1, is_file=1):
        if msg[-1] != "\n": msg = msg + "\n"
        if '\r' in msg: is_file = 0
        if is_terminal == 1:
            self.terminal.write(msg)
            self.terminal.flush()
        if is_file == 1 and not self.no_save:
            self.file.write(msg)
            self.file.flush()
    def flush(self): 
        pass

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.write('{:25s}: {}\n'.format(k, v))
        else:
            print('{:25s}: {}'.format(k, v))

def save_args(args, to_path):
    with open(to_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

@torch.no_grad()
def accuracy(net, loader):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc='Test...', leave=False) :
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total
