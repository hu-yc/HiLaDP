import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100, ImageFolder

from settings import args


def one_hot_encode(target):
    return torch.nn.functional.one_hot(torch.tensor(target), num_classes=args.nr_class)


def load_raw_data(dataset_name):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    train_set = CIFAR10(root=args.data_path, train=True, download=True,
                        transform=train_transform, target_transform=one_hot_encode
                        )
    val_set = CIFAR10(root=args.data_path, train=True, download=True,
                        transform=test_transform, target_transform=one_hot_encode
                        )
    test_set = CIFAR10(root=args.data_path, train=False, download=True,
                        transform=test_transform, target_transform=one_hot_encode
                        )
    return train_set, val_set, test_set


def load_disk_data(ds_name):
    train_set, _, test_set = load_raw_data(ds_name)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    return train_loader, test_loader

