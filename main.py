from __future__ import division

import argparse
import math
import os
import random
import shutil
import sys
import time
import pandas as pd
import numpy
from scipy.io import savemat
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_
from models.SimpleViT import get_vit_model
import torch
sys.path.append(".")
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.distributed.algorithms import ddp_comm_hooks

import communication_hook.hooks_JointSQ as myhooks

import models
from data.datasets import get_dataloader, get_ucml_dataloader, get_nwpu_dataloader
from optimizer.lamb import Lamb
from utils import AverageMeter, RecorderMeter, print_log

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
print(model_names)

def set_global_parm() -> None:
    global arg_global
    arg_global = get_args()
    global device_to_use
    device_to_use = int(dist.get_rank())
    print('===>>> use device {} <<<==='.format(device_to_use))
    global with_grad_compressed
    with_grad_compressed = arg_global.with_gc
    global with_params_sync
    with_params_sync = arg_global.params_sync
    
def get_args() -> argparse.Namespace:
    """parser args"""
    parser = argparse.ArgumentParser(description='Trains with compression',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset options
    parser.add_argument('--data_dir', type=str, help='Path to dataset', default='./data/cifar')
    parser.add_argument('--dataset', type=str, metavar='NAME', choices=['CIFAR10', 'CIFAR100', 'ImageNet', 'UCML', 'NWPU'],
                        help='Choose between CIFAR10/100 and ImageNet.', default='CIFAR10')

    # DDP input
    parser.add_argument('--local_rank', type=str, default='0')
    parser.add_argument('--seed', type=str, default='1234')
    parser.add_argument('--nproc_per_node', type=str, default='1')
    parser.add_argument('--nnode', type=str, default='1')
    parser.add_argument('--node_rank', type=str, default='0')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1')
    parser.add_argument('--master_port', type=str, default='12345')

    # The path of files to save
    parser.add_argument('--save_dir', type=str, default='./result/', help='Folder to save checkpoints and log.')
    parser.add_argument('--save_model', action='store_true', default=False, help='save model params')

    # Model options
    parser.add_argument('--arch', type=str, metavar='ARCH', default='resnet20', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet20)')

    # Optimization options
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='The Initial Learning Rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay (L2 penalty).')

    # Checkpoints
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set',
                        default=False)

    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')

    # Random seed
    parser.add_argument('--manualSeed', type=int, help='manual seed', default=None)
    parser.add_argument('--logSeed', type=str, help='log seed', default='test')

    # Compress parameters
    parser.add_argument('--with_gc', action='store_true', default=False, help='with grad compressed')
    parser.add_argument('--params_sync', action='store_true', default=False,
                        help='synchronize parameters once each epoch')

    # pretrain model
    parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', default=False,
                        help='use state dcit or not')
    parser.add_argument('--use_pretrain', dest='use_pretrain', action='store_true', default=False,
                        help='use pre-trained model or not')
    parser.add_argument('--pretrain_path', default='', type=str, help='..path of pre-trained model')

    return parser.parse_args()

def set_seed(seed: any, useCuda: bool, log) -> None:
    """set seed"""
    if seed is None:
        seed = random.randint(0, 2 ** 32)
    print_log('===>>> random seed : {}'.format(seed), log)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if useCuda:
        torch.cuda.manual_seed_all(seed)


def init_logger(args):
    """Init logger"""
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, 'log_rank{}_seed_{}.txt'.format(dist.get_rank(), args.logSeed)), 'w')
    print_log('===>>> save path : {}'.format(args.save_dir), log)
    print_log("===>>> torch  version : {}".format(torch.__version__), log)
    print_log("===>>> cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("===>>> use pretrain: {}".format(args.use_pretrain), log)
    if args.use_pretrain:
        print_log("===>>> Pretrain path: {}".format(args.pretrain_path), log)

    return log


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(model, val_loader, criterion, log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    timepoint = time.time()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(val_loader, desc='test', ncols=0, disable=(dist.get_rank() != 0))):
            input_var = input.to(device_to_use)
            target_var = target.to(device_to_use)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
            test_loss = loss.item()
            losses.update(test_loss, input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - timepoint)
            timepoint = time.time()

    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def sync_params(model):  # Synchronize model parameters, can be set to synchronize every multiple epochs.
    for _, param in enumerate(model.parameters()):
        dist.all_reduce(param.data.div_(dist.get_world_size()), op=dist.ReduceOp.SUM, async_op=False)


def train_epoch(model, epoch_curr, train_sampler, train_loader, arg, criterion, optimizer, scheduler, log,
                *args, **kwargs):
    # global g1, g2
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    # In distributed mode, calling the set_epoch() method at the beginning of each epoch before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used.
    train_sampler.set_epoch(time.time_ns())
    for step, (input, target) in enumerate(tqdm(train_loader, desc='train', ncols=0, disable=(dist.get_rank() != 0))):
        # if arg.use_cuda:
        input_var, target_var, model = input.to(device_to_use, non_blocking=True), \
            target.to(device_to_use, non_blocking=True), \
            model.to(device_to_use, non_blocking=True)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
        losses.update(loss.item(), input_var.size(0))
        top1.update(prec1[0], input_var.size(0))
        top5.update(prec5[0], input_var.size(0))
        
        # compute gradient and do SGD step
        loss.backward()

        clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)  # Gradient clipping
        optimizer.step()
        optimizer.zero_grad()

        # Synchronize the parameters 
        if arg_global.params_sync and step % 100 == 0:
            sync_params(model)

    # adjust the learning rate
    scheduler.step()
    print_log('[rank: {} | epoch: {}/{} | loss: {}]'.format(dist.get_rank(), epoch_curr + 1, arg.epochs, losses.avg),
              log)


def train(arg, model, optimizer, train_sampler, train_loader, criterion, val_loader, log, *args, **kwargs):
    best_prec1 = 0
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = arg.epochs)
    if arg.epochs == 100:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    elif arg.epochs == 500:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.2, milestones=[60, 120, 160, 190])
    elif arg.epochs == 200:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    # with_grad_compressed = 1 means compression / with_grad_compressed = 0 means no compression
    # Communication hook: Registered when loss.backward is called.
    if with_grad_compressed:
        ddp_comm_hooks.register_ddp_comm_hook(myhooks.DDPCommHookType.correlation_GC, model) 
        print_log('===>>>with_GC<<<===', log)
    else:
        ddp_comm_hooks.register_ddp_comm_hook(ddp_comm_hooks.DDPCommHookType.ALLREDUCE, model)
        print_log('===>>>without_GC<<<===', log)


    for epoch_curr in range(arg.start_epoch, arg.epochs):  # 100
        # train for one epoch
        s_time = time.time()
        train_epoch(model, epoch_curr, train_sampler, train_loader, arg, criterion, optimizer, scheduler, log)
        e_time = time.time()

        # Calculate test set accuracy
        val_acc_2, test_loss = validate(model, val_loader, criterion, log)

        # remember best prec@1 and save checkpoint
        is_best = val_acc_2 > best_prec1
        best_prec1 = max(val_acc_2, best_prec1)
        if arg.save_model:
            save_checkpoint({
                'epoch': epoch_curr + 1,
                'arch': arg.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, os.path.join(arg.save_dir, 'resnet20-cifar10-gc' + str(arg.rate_gc) + '.pth'),
                os.path.join(arg.save_dir,
                             'best.resnet20-cifar10-gc' + str(arg.manualSeed) + str(arg.rate_gc) + '.pth'))
        print_log(
            '\033[0;37;40m|------[rank {} on device {} | epoch: {}/{} | acc/test_top_1: {:.3f}]------|\033[0m'.format(
                dist.get_rank(), device_to_use, epoch_curr + 1, arg.epochs, val_acc_2.cpu().numpy()), log)
        
        list = [dist.get_rank(), device_to_use, epoch_curr + 1, e_time - s_time, best_prec1.cpu().numpy(),test_loss,val_acc_2.cpu().numpy()]
        data = pd.DataFrame([list])
        # save_path
        data.to_csv('vit.csv',mode='a',header=False,index=False)

        print_log(
            '\033[0;31;40m|------[rank {} on device {} | epoch: {}/{} | cost: {:.3f}s | acc/test_top1_best: {:.3f}]------|\033[0m'.format(
                dist.get_rank(), device_to_use, epoch_curr + 1, arg.epochs, e_time - s_time, best_prec1.cpu().numpy()),
            log)

def main():
    args = arg_global
    args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()
    cudnn.benchmark = False  # This can slow down training
    log = init_logger(args)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    set_seed(args.manualSeed, args.use_cuda, log)

    # Data
    print_log('==>>> Preparing data..', log)
    if not os.path.isdir(args.data_dir):
        os.makedirs(args.data_dir)
    if args.dataset == 'CIFAR10':
        num_cls = 10
        img_size = 32
    elif args.dataset == 'CIFAR100':
        num_cls = 1020
        img_size = 32
    elif args.dataset == 'ImageNet':
        num_cls = 1000
        img_size = 224
    elif args.dataset == 'UCML':
        num_cls = 21
        img_size = 256
    elif args.dataset == 'NWPU':
        num_cls = 45
        img_size = 256
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'UCML':
        train_loader, val_loader, test_loader, train_sampler = get_ucml_dataloader(args.batch_size)
    elif args.dataset == 'NWPU':
        train_loader, val_loader, test_loader, train_sampler = get_nwpu_dataloader(args.batch_size)
    else:
        train_loader, val_loader, test_loader, train_sampler = get_dataloader(img_size, args.dataset, args.data_dir,

                                                                                  args.batch_size, no_val=True)

    # Init model
    #print_log("==>>> creating model '{}'".format(args.arch), log)
    # Note: Here, calling SimpleViT executes model = get_vit_model()
    # For other models, execute: model = models.__dict__[args.arch](num_classes=num_cls)
    # model = get_vit_model()
    model = models.__dict__[args.arch](num_classes=num_cls)
    print_log("==>>> model :\n {}".format(model), log)

    criterion = torch.nn.CrossEntropyLoss()

    if args.use_cuda:
        model = model.cuda(device_to_use)
        criterion = criterion.cuda(device_to_use)

    if args.use_pretrain:
        if os.path.isfile(args.pretrain_path):
            print_log("===>>> loading pretrain model '{}'".format(args.pretrain_path), log)
        else:
            print("\033[1;33;40m!!! pretrain path does not exist !!!\033[0m")
            pass
        pretrain = torch.load(args.pretrain_path)
        if args.use_state_dict:
            model.load_state_dict(pretrain['state_dict'])
        else:
            model = pretrain['state_dict']

    recorder = RecorderMeter(args.epochs)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_to_use], output_device=device_to_use,
                                                      broadcast_buffers=False, bucket_cap_mb=25)

    optimizer = torch.optim.SGD(model.parameters(), state['lr'] , momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=False)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print_log("===>>> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            if args.use_state_dict:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model = checkpoint['state_dict']

            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("===>>> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("===>>> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("===>>> do not use any checkpoint for {} model <<<===".format(args.arch), log)

    if args.evaluate:
        time1 = time.time()
        validate(model, test_loader, criterion, args.print_freq, log)
        time2 = time.time()
        print('function took %0.3f ms' % ((time2 - time1) * 1000.0))
        return

    val_acc_1, _ = validate(model, test_loader, criterion, log)
    print("===>>> acc before is: %.3f %% <<<===" % val_acc_1)

    # NOTE train
    train(args, model, optimizer, train_sampler, train_loader, criterion, test_loader, log)

    log.close()


if __name__ == "__main__":
    # Set visible GPUs
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
    # Set the network interface used by local processes (not needed for single machine)
    # os.environ['NCCL_SOCKET_IFNAME'] = 'eth1'
    # DDP initialization
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        group_name='test'
    )
    print('===>>> DDP init successfully <<<===')
    print('===>>> rank is {} <<<==='.format(dist.get_rank()))
    set_global_parm()
    main()
