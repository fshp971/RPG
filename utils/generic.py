import torch
import torchvision.transforms as transforms

import models
from . import data
from . import datasets

class AverageMeter():
    def __init__(self):
        self.cnt = 0
        self.sum = 0
        self.mean = 0

    def update(self, val, cnt):
        self.cnt += cnt
        self.sum += val * cnt
        self.mean = self.sum / self.cnt

    def average(self):
        return self.mean
    
    def total(self):
        return self.sum


def add_log(log, key, value):
    if key not in log.keys():
        log[key] = []
    log[key].append(value)


def get_arch(arch, dataset):
    if dataset == 'cifar10':
        in_dims, out_dims, wide = 3, 10, 10
    elif dataset == 'cifar100':
        in_dims, out_dims, wide = 3, 100, 10
    else:
        raise NotImplementedError

    if arch == 'wrn':
        if 'cifar' in dataset:
            return models.WRN34_10(in_dims, out_dims)
    elif arch == 'wrn-50-2':
        return models.WRN50_2(in_dims, out_dims)
    else:
        raise NotImplementedError


def get_loader(dataset, batch_size, train=True, training=True):
    if dataset == 'cifar10':
        in_size, in_dims, out_dims, padding = 32, 3, 10, 4
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.))

    elif dataset == 'cifar100':
        in_size, in_dims, out_dims, padding = 32, 3, 100, 4
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (1., 1., 1.))

    else:
        raise NotImplementedError

    if training:
        compose = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(in_size, padding),
            transforms.ToTensor(), normalize,]
    else:
        compose = [transforms.ToTensor(), normalize,]

    transform = transforms.Compose( compose )


    if dataset == 'cifar10':
        target_set = datasets.CIFAR10('./data', train=train, transform=transform)
    elif dataset == 'cifar100':
        target_set = datasets.CIFAR100('./data', train=train, transform=transform)
    else:
        raise NotImplementedError

    if training:
        loader = data.TrainLoader(target_set, batch_size)
    else:
        loader = data.EvalLoader(target_set, batch_size)

    return loader
