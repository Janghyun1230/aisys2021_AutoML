import torch
import os, sys, shutil, time, random
from torch.autograd import Variable
from existing_models.load_data import load_data_subset
from existing_models.logger import plotting, copy_script_to_folder, AverageMeter, RecorderMeter, time_string, convert_secs2time

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, optimizer, device, loss_fn, eval_fn):
    '''train given model and dataloader'''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    if device == "cuda":
        bce_loss = torch.nn.BCELoss().cuda()
        softmax = torch.nn.Softmax(dim=1).cuda()
    else:
        bce_loss = torch.nn.BCELoss()
        softmax = torch.nn.Softmax(dim=1)

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        if device == "cuda":
            input = input.cuda()
            target = target.long().cuda()
        else:
            input = input
            target = target.long()

        # train with clean images
        input_var, target_var = Variable(input), Variable(target)
        output, reweighted_target = model(input_var, target_var)
        loss = loss_fn(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = eval_fn(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(
        '  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
        .format(top1=top1, top5=top5, error1=100 - top1.avg))

    return top1.avg, top5.avg, losses.avg


def validate(val_loader,
             model,
             device,
             loss_fn,
             eval_fn,
             fgsm=False,
             eps=4,
             rand_init=False,
             mean=None,
             std=None):
    '''evaluate trained model'''
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if device == "cuda":
            input = input.cuda()
            target = target.cuda()

        with torch.no_grad():
            input_var = Variable(input)
            target_var = Variable(target)

            # compute output
            output = model(input_var)
            loss = loss_fn(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = eval_fn(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

    print(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f} Loss: {losses.avg:.3f} '
            .format(top1=top1, top5=top5, error1=100 - top1.avg,
                    losses=losses))

    print(top1.avg, top5.avg)
    return top1.avg, top5.avg, losses.avg
