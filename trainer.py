import torch
import time
from torch.autograd import Variable
from logger import AverageMeter


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

    for input, target in train_loader:
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        input = input.to(device)
        target = target.long().to(device)

        # train with clean images
        output = model(input)
        loss = loss_fn(output, target)

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

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, device, loss_fn, eval_fn):
    '''evaluate trained model'''
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for input, target in val_loader:
        input = input.to(device)
        target = target.long().to(device)

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

    return top1.avg, top5.avg, losses.avg
