import os, time, torch, torchvision
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
# from filelock import FileLock
import torch.multiprocessing as mp
from torchvision.datasets import CIFAR10, CIFAR100


# Dataloader
def get_dataloader(img_size, dataset, datapath, batch_size, no_val):
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # 用均值和标准差对张量图像进行归一化
    WORKS = 0
    
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        train_set = eval(dataset)(datapath, True, torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(img_size, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]), download=True)
        val_set = eval(dataset)(datapath, True, torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                normalize,
            ]), download=True)

        test_set = eval(dataset)(datapath, False, torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                normalize,
            ]), download=True)

        distributed_train_sampler = DistributedSampler(
            train_set, 
            num_replicas=dist.get_world_size(), 
            rank = dist.get_rank(), 
            shuffle=True
            )

        distributed_val_sampler = DistributedSampler(
            val_set,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )

        distributed_test_sampler = DistributedSampler(
            test_set,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False
        )

        if no_val:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, 
                sampler=distributed_train_sampler,
                num_workers=WORKS, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, sampler=distributed_val_sampler,
                num_workers=WORKS, pin_memory=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size,
                sampler=distributed_train_sampler,
                num_workers=WORKS, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, sampler=distributed_val_sampler,
                num_workers=WORKS, pin_memory=True
            )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size,
            # sampler=distributed_test_sampler,
            num_workers=WORKS, pin_memory=False
        )

    elif dataset == 'ImageNet':
        # train_set = datasets.ImageNet(datapath, split='train', download=False, transform=torchvision.transforms.Compose([transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))

        # val_set = datasets.ImageNet(datapath, split='train', download=False, transform=torchvision.transforms.Compose([transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))

        # test_set = datasets.ImageNet(datapath, split='val', download=False, transform=torchvision.transforms.Compose([transforms.Resize(int(img_size / 0.875)), transforms.CenterCrop(img_size), transforms.ToTensor(), normalize]))

        train_path = datapath + '/train'
        val_path = datapath + '/val' 

        train_set = datasets.ImageFolder(train_path, transform=torchvision.transforms.Compose([transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))

        val_set = datasets.ImageFolder(train_path, transform=torchvision.transforms.Compose([transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))

        test_set = datasets.ImageFolder(val_path, transform=torchvision.transforms.Compose([transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor(), normalize]))

        distributed_train_sampler = DistributedSampler(
            train_set, 
            num_replicas=dist.get_world_size(), 
            rank = dist.get_rank(), 
            shuffle=True
            )
        
        distributed_val_sampler = DistributedSampler(
            val_set,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )

        distributed_test_sampler = DistributedSampler(
            test_set,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, 
            sampler=distributed_train_sampler,
            num_workers=WORKS, pin_memory=True
            )

        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, 
            sampler=distributed_val_sampler,
            num_workers=WORKS, pin_memory=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size,
            sampler=distributed_test_sampler,
            num_workers=WORKS, pin_memory=False
        )

    return train_loader, val_loader, test_loader, distributed_train_sampler

def get_ucml_dataloader(batch_size):
    WORKS = 0

    ROOT_TRAIN = '/home/wangzixuan/data/UCMerced_LandUse/train'
    ROOT_TEST = '/home/wangzixuan/data/UCMerced_LandUse/test'
    ROOT_VAL = '/home/wangzixuan/data/UCMerced_LandUse/val'

    # 将图像的像素值归一化到[-1， 1]之间
    normalize = transforms.Normalize(mean=[0.48422759, 0.49005176, 0.45050278],
                                std=[0.17348298, 0.16352356, 0.15547497])
    # 将训练集输入图像进行预处理，重映射尺寸、随机旋转、转化张量
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    # 将验证集输入图像进行预处理，重映射尺寸、随机旋转、转化张量
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize])
    # ////
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize])
    # 设置训练集、验证集加载路径
    train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
    val_dataset = ImageFolder(ROOT_VAL, transform=val_transform)
    test_dataset = ImageFolder(ROOT_TEST, transform=test_transform)

    distributed_train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )

    distributed_val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )

    distributed_test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False
    )

    # 加载训练集、验证集
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=distributed_train_sampler,
        # shuffle=True,
        num_workers=WORKS, pin_memory=True)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        sampler=distributed_val_sampler,
        # shuffle=True,
        num_workers=WORKS, pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        # sampler=distributed_test_sampler,
        shuffle=True, num_workers=WORKS, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, distributed_train_sampler


def get_nwpu_dataloader(batch_size):
    WORKS = 0

    ROOT_TRAIN = '/home/wangzixuan/data/NWPU/train'
    ROOT_TEST = '/home/wangzixuan/data/NWPU/test'
    ROOT_VAL = '/home/wangzixuan/data/NWPU/val'

    # 将图像的像素值归一化到[-1， 1]之间
    normalize = transforms.Normalize(mean=[0.48422759, 0.49005176, 0.45050278],
                                std=[0.17348298, 0.16352356, 0.15547497])
    normalize = transforms.Normalize(mean=[0.36801905, 0.3809775, 0.34357441],
                                std=[0.14530348, 0.13557449, 0.13204114])

    # 将训练集输入图像进行预处理，重映射尺寸、随机旋转、转化张量
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    # 将验证集输入图像进行预处理，重映射尺寸、随机旋转、转化张量
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize])
    # ////
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        normalize])
    # 设置训练集、验证集加载路径
    train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
    val_dataset = ImageFolder(ROOT_VAL, transform=val_transform)
    test_dataset = ImageFolder(ROOT_TEST, transform=test_transform)

    distributed_train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )

    distributed_val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True
    )

    distributed_test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False
    )

    # 加载训练集、验证集
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=distributed_train_sampler,
        # shuffle=True,
        num_workers=WORKS, pin_memory=True)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        sampler=distributed_val_sampler,
        # shuffle=True,
        num_workers=WORKS, pin_memory=True)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        # sampler=distributed_test_sampler,
        shuffle=True, num_workers=WORKS, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader, distributed_train_sampler