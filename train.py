import sys
from os.path import abspath, dirname

sys.path.append(abspath(dirname(__file__)))

import argparse
import os
import os.path as osp
import shutil
import time
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from framework.datasets import build_transform, build_dataset
from framework.modeling.backbone import build_model
from framework.modeling.model_builder import NaiveModelBuilder, ModelBuilder
from framework.modeling.kd import build_kd_model
from framework.losses import build_criterion
from framework.utils.meters import AverageMeter
from framework.utils.accuracy import accuracy
from framework.utils.logging import Logger


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('-j', '--workers', default=1, type=int, help='number of data loading workers (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--teacher_model',
                    default='', type=str, help='path to teacher model (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--normalized', action="store_true", default=False)
parser.add_argument('--kd_method', type=str, default='kd',
                    help='The name of the knowledge distillation methods.')
parser.add_argument('--alpha', nargs='+', type=float, default=[0.8],
                    help='Hypterparameters for the loss function.')
parser.add_argument('--temperature', type=int, default=16,
                    help='The temperature used in the KD paper.')
parser.add_argument('--preReLU', action="store_true", default=False,
                    help='Whether to use feature maps before the ReLU activation, mentioned in the MARGIN paper.')
parser.add_argument('--teacher_name', type=str)
parser.add_argument('--student_name', type=str)
parser.add_argument('--root', type=str, default="")

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    sys.stdout = Logger(osp.join('runs/{}'.format(args.name), 'log.txt'))
    if args.tensorboard:
        train_logger = SummaryWriter(osp.join('runs/{}'.format(args.name), 'train'))
        val_logger = SummaryWriter(osp.join('runs/{}'.format(args.name), 'val'))
    else:
        train_logger = val_logger = None

    # Data loading code
    train_data = build_dataset(
        dict(name=args.dataset, transform=build_transform(args.dataset, training=True), training=True,
             root=osp.join(args.root, 'train')))
    test_data = build_dataset(
        dict(name=args.dataset, transform=build_transform(args.dataset, training=False), training=False,
             root=osp.join(args.root, 'test')))

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, **kwargs)

    # create model
    model_s = build_model(dict(name=args.student_name, args=args, num_classes=train_data.num_classes))
    model_t = build_model(dict(name=args.teacher_name, args=args, num_classes=train_data.num_classes))

    size_s = model_s.get_size(train_data.img_size)
    size_t = model_t.get_size(train_data.img_size)
    margin_t = model_t.get_margin()
    model_kd = build_kd_model(dict(name=args.kd_method, size_s=size_s, size_t=size_t, margin_t=margin_t))

    if args.kd_method == 'kd':
        model = NaiveModelBuilder(model_s, model_t)
    else:
        model = ModelBuilder(model_s, model_t, model_kd, args.normalized, args.preReLU)

    if os.path.isfile(args.teacher_model):
        print("=> loading teacher model from '{}'".format(args.teacher_model))
        teacher_checkpoint = torch.load(args.teacher_model)
        model.model_t.load_state_dict(teacher_checkpoint['state_dict'])
    else:
        print(f"fail to find teacher model on {args.teacher-model}")

    # get the number of model parameters
    print('Number of teacher model parameters: {}'.format(
        sum([p.data.nelement() for p in model_t.parameters()])))
    print('Number of student model parameters: {}'.format(
        sum([p.data.nelement() for p in model_s.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    device = torch.device("cuda")
    model = model.to(device)
    model = model.cuda()
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    train_criterion = build_criterion(dict(name=args.kd_method, margin_t=margin_t, args=args)).cuda()
    test_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)

    # learning rate scheduler
    if args.dataset == 'cinic10':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 120], gamma=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, model, train_criterion, optimizer, scheduler, epoch, train_logger)
        scheduler.step()

        # evaluate on validation set
        prec1 = validate(test_loader, model, test_criterion, epoch, val_logger)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_s.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)
    print('Best accuracy: ', best_prec1)


def train(train_loader, model, criterion, optimizer, scheduler, epoch, logger):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    model.model_t.eval()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        if args.kd_method == 'kd' or args.kd_method == 'new_kd':
            output_s, output_t = model(input)
            loss = criterion(output_s, output_t, target)
        else:
            output_s, mat_s, mat_t = model(input)
            loss = criterion(output_s, target, mat_s, mat_t)
        # print(loss)
        # measure accuracy and record loss
        prec1 = accuracy(output_s.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.kd_method == 'new_kd':
            criterion.step()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        logger.add_scalar('loss', losses.avg, epoch)
        logger.add_scalar('acc', top1.avg, epoch)


def validate(val_loader, model, criterion, epoch, logger):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        logger.add_scalar('loss', losses.avg, epoch)
        logger.add_scalar('acc', top1.avg, epoch)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.name) + 'model_best.pth.tar')


if __name__ == '__main__':
    main()
