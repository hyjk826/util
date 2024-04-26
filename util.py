import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import argparse

args = argparse.ArgumentParser()

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
