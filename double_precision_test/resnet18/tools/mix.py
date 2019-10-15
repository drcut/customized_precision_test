import numpy as np
import math
import argparse
import time
import yaml
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.distributed import broadcast
import torchvision.transforms as transforms
import models
from tensorboardX import SummaryWriter
from utils.train_util import AverageMeter, accuracy, save_checkpoint, load_state, IterLRScheduler, DistributedGivenIterationSampler, DistributedSampler, simple_group_split
from utils.dist_util import dist_init, sum_gradients, DistModule
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
torch.manual_seed(24)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/res50.yaml')
parser.add_argument('--dist', action='store_true', help='distributed training or not')
parser.add_argument('--double', action='store_true')
parser.add_argument('--half', action='store_true')
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--resume-opt', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--loss_scale', type=float, default=32,
                    help='loss scale help fp16 convergence')
args = parser.parse_args()

rank = 0
world_size = 1
best_prec1 = 0.
dataset_len = None
emulate_node = 2

def prep_param_lists(model, flat_master=False):
    global args
    model_params = [param for param in model.parameters() if param.requires_grad]

    if args.double:
        master_params = [param.clone().double().detach() for param in model_params]
    else:
        master_params = [param.clone().float().detach() for param in model_params]

    for param in master_params:
        param.requires_grad = True

    return model_params, master_params

def main():
    global args, rank, world_size, best_prec1, dataset_len, emulate_node

    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in config['common'].items():
        setattr(args, k, v)
    
    if args.dist:
        rank, world_size = dist_init()
    else:
        rank = 0
        world_size = 1
        print('Disabled distributed training.')

    # create model
    model = models.__dict__[args.arch]()

    model.cuda()

    if args.double:
        model.double()
    if args.half:
        model.half()
        def BN_convert_float(module):
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module = module.float()
            for child in module.children():
                BN_convert_float(child)
        BN_convert_float(model)

    
    if args.dist:
        for param in model.parameters():
            broadcast(param, 0)

    global model_params, master_params
    model_params, master_params = prep_param_lists(model)
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(master_params, args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    last_iter = -1
    if args.load_path:
        if args.resume_opt:
            best_prec1, last_iter = load_state(args.load_path, model, optimizer=optimizer)
        else:
            load_state(args.load_path, model)

    lr_scheduler = IterLRScheduler(optimizer, args.lr_steps, args.lr_mults, last_iter=last_iter)

    # Data loading code
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                 train=True,
                                                 transform = transform_train)
    val_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                 train=False,
                                                 transform = transform_test)
    dataset_len = len(train_dataset)

    args.max_iter = math.ceil((dataset_len*args.max_epoch)/(world_size*args.batch_size*emulate_node))
    if args.dist:
        train_sampler = DistributedGivenIterationSampler(train_dataset, args.max_iter*emulate_node, args.batch_size, last_iter=last_iter)
        val_sampler = DistributedSampler(val_dataset, round_up=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if rank == 0:
        tb_logger = SummaryWriter(args.save_path)
    else:
        tb_logger = None

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    train(train_loader, val_loader, model, criterion, optimizer, lr_scheduler, last_iter + 1, tb_logger)

def adjust_learning_rate(optimizer, step):
    global dataset_len,rank, emulate_node

    iter_per_epoch = math.ceil(dataset_len/(world_size * args.batch_size *emulate_node))
    warm_up_iter = 5 * iter_per_epoch

    if(step<=warm_up_iter):
        lr = 0.1 + (1.6 - 0.1) * (step/warm_up_iter)
    else:
        lr = 1.6
        if step > iter_per_epoch * 40:
            lr *= 0.1
        #if step > iter_per_epoch * 70:
        #    lr *= 0.1
        if step > iter_per_epoch * 80:
            lr *= 0.1 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(train_loader, val_loader, model, criterion, optimizer, lr_scheduler, start_iter, tb_logger):

    global args, rank, world_size, best_prec1, emulate_node
    global grad_exp,grad_man,param_exp,param_man

    batch_time = AverageMeter(args.print_freq)
    data_time = AverageMeter(args.print_freq)
    losses = AverageMeter(args.print_freq)

    model.train()

    end = time.time()
    curr_step = start_iter
    emulate_step = 0

    for i, (input, target) in enumerate(train_loader):
        emulate_step += 1
        if emulate_step == emulate_node:
            curr_step += 1
        if curr_step > args.max_iter:
            break

        # lr_scheduler.step(curr_step)
        # current_lr = lr_scheduler.get_lr()[0]
        current_lr = adjust_learning_rate(optimizer, curr_step)

        target = target.cuda()
        input_var = input.cuda()
        if args.double:
            input_var = input_var.double()
        if args.half:
            input_var = input_var.half()

        data_time.update(time.time() - end)

        output = model(input_var,rank)
        loss = criterion(output, target) / (world_size*emulate_node)
        reduced_loss = loss.data.clone()
        if args.dist:  
            dist.all_reduce(reduced_loss)
        losses.update(float(reduced_loss.item()))
        loss = loss * args.loss_scale
        model.zero_grad()
        loss.backward()

        if args.dist:
            sum_gradients(model)

        for model_p,master_p in zip(model_params, master_params):
            if model_p.grad is not None:
                if args.double:
                    master_p.backward(model_p.grad.double()/args.loss_scale)
                else:
                    master_p.backward(model_p.grad.float()/args.loss_scale)

        if emulate_node == emulate_step:
            emulate_step = 0
            optimizer.step()
            param_list = []
            for model_p,master_p in zip(model_params, master_params):
                model_p.data.copy_(master_p.data)
            optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            if (curr_step == 1 or curr_step % args.print_freq == 0) and rank == 0:
                if tb_logger:
                    tb_logger.add_scalar('loss_train', losses.avg, curr_step)
                    tb_logger.add_scalar('lr', current_lr, curr_step)
                print('Iter: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'LR {lr:.4f}'.format(
                       curr_step, args.max_iter, batch_time=batch_time,
                       data_time=data_time, loss=losses, lr=current_lr))

            if curr_step % args.val_freq == 0 and curr_step != 0:
                val_loss, prec1, prec5 = validate(val_loader, model, criterion)
    
                if tb_logger:
                    tb_logger.add_scalar('loss_val', val_loss, curr_step)
                    tb_logger.add_scalar('acc1_val', prec1, curr_step)
                    tb_logger.add_scalar('acc5_val', prec5, curr_step)

                if rank == 0:
                    # remember best prec@1 and save checkpoint
                    is_best = prec1 > best_prec1
                    best_prec1 = max(prec1, best_prec1)
                    save_checkpoint({
                        'step': curr_step,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, args.save_path+'/ckpt_'+str(curr_step))
    val_loss, prec1, prec5 = validate(val_loader, model, criterion)

def validate(val_loader, model, criterion):

    global args, rank, world_size, best_prec1

    # validation don't need track the history
    batch_time = AverageMeter(args.print_freq)
    losses = AverageMeter(args.print_freq)
    top1 = AverageMeter()
    top5 = AverageMeter()

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
        if args.double:
            input = input.double()
        if args.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            output = model(input, -1)

        # measure accuracy and record loss
        loss = criterion(output, target) / world_size
        prec1, prec5 = accuracy(output.float().data, target, topk=(1, 5))

        reduced_loss = loss.data.clone()
        reduced_prec1 = prec1.clone() / world_size
        reduced_prec5 = prec5.clone() / world_size
        
        if args.dist:
            dist.all_reduce(reduced_loss)
            dist.all_reduce(reduced_prec1)
            dist.all_reduce(reduced_prec5)

        losses.update(reduced_loss.item())
        top1.update(reduced_prec1.item())
        top5.update(reduced_prec5.item())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and rank == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    if rank == 0:
        print(' * All Loss {loss.avg:.4f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(loss=losses, top1=top1, top5=top5))

    model.train()

    return losses.avg, top1.avg, top5.avg

if __name__ == '__main__':
    main()
