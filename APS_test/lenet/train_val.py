import argparse
import time
import math

import torch
import torch.cuda
import torch.nn as nn
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from train_utils import AverageMeter, accuracy, DistributedGivenIterationSampler, DistributedSampler
from CPDtorch.utils.dist_util import dist_init, sum_gradients, DistModule

from lenet import LeNet


parser = argparse.ArgumentParser()
parser.add_argument('--dist', default=0,type=int)
parser.add_argument('--epoch', default=30,type=int)
parser.add_argument('--warm_up_epoch', default=5,type=int)
parser.add_argument('--base_lr', default=0.01,type=float)
parser.add_argument('--final_lr', default=0.01*128,type=float)
parser.add_argument('-b','--batch_size', default=256,type=int)
parser.add_argument('--momentum', default=0.9,type=float)
parser.add_argument('--weight_decay',default=0.01,type=float)
parser.add_argument('--workers',default=4)
parser.add_argument('--val_freq',default=5,type=int)
parser.add_argument('--print_freq',default=100,type=int)

args = parser.parse_args()

rank = 0
world_size = 1
best_prec1 = 0.
dataset_len = None

def main():
    global args, rank, world_size, best_prec1, dataset_len
    
    if args.dist == 1:
        rank, world_size = dist_init()
    else:
        rank = 0
        world_size = 1

    model = LeNet()
    model.cuda()

    param_copy = [param.clone().type(torch.cuda.FloatTensor).detach() for param in model.parameters()]

    for param in param_copy:
        param.requires_grad = True

    if args.dist == 1:
        model = DistModule(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(param_copy, args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    last_iter = -1

    # Data loading code
    train_dataset = datasets.MNIST(root = './data',
                                train=True,
                                transform = transforms.ToTensor(),
                                download=False)
    val_dataset = datasets.MNIST(root = './data',
                                train=False,
                                transform=transforms.ToTensor(),
                                download=False)

    dataset_len = len(train_dataset)
    args.max_iter = math.ceil((dataset_len * args.epoch)/(world_size * args.batch_size))

    if args.dist == 1:
        train_sampler = DistributedGivenIterationSampler(train_dataset, args.max_iter, args.batch_size, last_iter=last_iter)
        val_sampler = DistributedSampler(val_dataset, round_up=False)
    else:
        train_sampler = DistributedGivenIterationSampler(train_dataset, args.max_iter, args.batch_size, world_size=1, rank=0, last_iter=last_iter)
        val_sampler = None

    # pin_memory if true, will copy the tensor to cuda pinned memory
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    train(train_loader, val_loader, model, criterion, optimizer,param_copy)

def adjust_learning_rate(optimizer, step):
    global dataset_len,rank
    warm_up_iter = math.ceil((args.warm_up_epoch * dataset_len)/(world_size * args.batch_size))
    if(step<=warm_up_iter):
        lr = args.base_lr + (args.final_lr - args.base_lr) * (step/warm_up_iter)
    else:
        lr = args.final_lr 
    lr = lr * (1 - step/args.max_iter)**2
    return lr


def train(train_loader, val_loader, model, criterion, optimizer,param_copy):

    global args, rank, world_size, best_prec1,dataset_len

    # moving average batch time, data loading time, loss
    batch_time = AverageMeter(10)
    data_time = AverageMeter(10)
    losses = AverageMeter(10)

    model.train()

    end = time.time()
    curr_step = 0

    momentum_buffer = []
    for master_p in param_copy:
        momentum_buffer.append(torch.zeros_like(master_p))

    for i, (input, target) in enumerate(train_loader):
        curr_step += 1
        if curr_step > args.max_iter:
            break

        current_lr = adjust_learning_rate(optimizer,curr_step)

        target = target.cuda()
        input = input.cuda()

        data_time.update(time.time() - end)
        end = time.time()

        output = model(input)
        # loss divided by world_size
        loss = criterion(output, target) / world_size
        if rank == 0:
            print("loss:",loss)

        # average loss
        reduced_loss = loss.data.clone()
        if args.dist == 1:  
            dist.all_reduce(reduced_loss)
        losses.update(reduced_loss.item())

        # average gradient
        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()

        if args.dist == 1:
            sum_gradients(model)
        
        for param_1,param_2 in zip(param_copy,list(model.parameters())):
            param_1.backward(param_2.grad.float())

        for idx, master_p in enumerate(param_copy):
            if master_p.grad is not None:
                update = master_p.grad
                local_lr = master_p.norm(2)/\
                            (master_p.grad.norm(2) \
                             + args.weight_decay * master_p.norm(2))
                momentum_buffer[idx] = args.momentum * momentum_buffer[idx] \
                                        + current_lr \
                                          * local_lr \
                                          * (master_p.grad + args.weight_decay * master_p)
                update = momentum_buffer[idx]
                master_p.data.copy_(master_p - update)

        for param,copy_param in zip(model.parameters(),param_copy):
            param.data.copy_(copy_param.data)

        batch_time.update(time.time() - end)
        end = time.time()
        if curr_step % args.val_freq == 0 and curr_step != 0:
            if rank == 0:
                print('Iter: [{}/{}]\nTime {batch_time.val:.3f} ({batch_time.avg:.3f})\n'.format(curr_step,args.max_iter,batch_time=batch_time))
            val_loss, prec1, prec5 = validate(val_loader, model, criterion)

    if rank == 0:
        print('Iter: [{}/{}]\n'.format(curr_step,args.max_iter))
    val_loss, prec1, prec5 = validate(val_loader, model, criterion)


def validate(val_loader, model, criterion):

    global args, rank, world_size, best_prec1

    # validation don't need track the history
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)
    top1 = AverageMeter(0)
    top5 = AverageMeter(0)

    # switch to evaluate mode
    model.eval()

    c1 = 0
    c5 = 0
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if i == len(val_loader) / (args.batch_size * world_size):
            break

        input = input.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(input)

        # measure accuracy and record loss
        loss = criterion(output, target) / world_size
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        reduced_loss = loss.data.clone()
        reduced_prec1 = prec1.clone() / world_size
        reduced_prec5 = prec5.clone() / world_size
        
        if args.dist == 1:
            dist.all_reduce(reduced_loss)
            dist.all_reduce(reduced_prec1)
            dist.all_reduce(reduced_prec5)

        losses.update(reduced_loss.item())
        top1.update(reduced_prec1.item())
        top5.update(reduced_prec5.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if rank == 0:
        print(' * All Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(loss=losses, top1=top1, top5=top5))

    model.train()

    return losses.avg, top1.avg, top5.avg

if __name__ == '__main__':
    main()
