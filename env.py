import argparse
import torch
import os
import logger
from time import time
import models
from trainer import train, validate, accuracy
from data import dataloader


def arg2model(inp_arg, device='cuda', from_pretrained=False):
    w, d, r = inp_arg
    path = "./" + str(w) + "_" + str(d) + "_" + str(r)
    if os.path.exists(path):
        print("Load existing model: " + path)
        model = torch.load(path)
    elif from_pretrained:
        print("Currently, no pretrained weight")
    else:
        print("Create new model")
        model = models.__dict__['preactresnet18'](100, False, 1, w,
                                                  d).to(device)
    return model


def _print_model(model, resolution, device='cuda'):
    report, _ = logger.summary_string(model, (3, resolution, resolution),
                                      batch_size=1,
                                      device=device)
    print(report)


class NasEnv():
    def __init__(self,
                 batch_size=64,
                 lr=1e-2,
                 resolution=32,
                 n_train=45000,
                 n_epoch=30):
        self.trainloader, self.validloader, self.testloader = dataloader(
            batch_size=batch_size, input_resolution=resolution)
        self.batch_size = batch_size
        self.resolution = resolution
        self.lr = lr
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.eval_fn = accuracy
        self.epoch = n_epoch

    def eval_arg(self, inp_arg, print_model=False):
        print("\nStart evaluating ", inp_arg)
        model = arg2model(inp_arg, device=self.device)
        if print_model:
            _print_model(model, self.resolution, device=self.device)
        mem = self.eval_mem(model)
        acc = self.eval_acc(model)
        latency = self.eval_latency(inp_arg)

        return mem, acc, latency

    def eval_mem(self, model):
        report, _ = logger.summary_string(
            model, (3, self.resolutio, self.resolution),
            batch_size=self.batch_size,
            device=self.device)

        estimated_mem = float(report.split('\n')[-3].split(' ')[-1])  # (MB)
        return estimated_mem

    def eval_latency(self, inp_arg):
        '''
        TODO
        '''
        return None

    def eval_acc(self, model):
        """
        Make model according to the input argument and return the accuracy and memory of the model.
        Args
        inp_arg: tuple(width_scale, depth_scale, resolution) e.g., (1.0, 1.0, 224)

        Return
        accuracy, memory of the model
        """
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=self.lr,
                                    momentum=0.9,
                                    weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.epoch // 2,
                                                    gamma=0.3)

        best_test_acc = 0

        for i in range(self.epoch):
            s = time()
            train_top1, train_top5, train_loss = self._train_epoch(
                model, optimizer)
            valid_top1, valid_top5, valid_loss = self._valid(model)

            if valid_top1 > best_test_acc:
                best_test_acc = valid_top1
            scheduler.step()
            print(
                f'[epoch {i} ({time()-s:.2f})] (train) loss {train_loss:.2f}, top1 {train_top1:.2f}%, top5 {train_top5:.2f}%',
                end=' | ')
            print(
                f'(valid) loss = {valid_loss:.2f}, top1 = {valid_top1:.2f}%, top5 = {valid_top5:.2f}%'
            )
        return best_test_acc

    def _train_epoch(self, model, optimizer):
        top1, top5, loss = train(self.trainloader, model, optimizer,
                                 self.device, self.loss_fn, self.eval_fn)
        return top1, top5, loss

    def _valid(self, model):
        top1, top5, loss = validate(self.validloader, model, self.device,
                                    self.loss_fn, self.eval_fn)
        return top1, top5, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''Environment'''
    parser.add_argument('-w', "--width", type=float, default=1.0)
    parser.add_argument('-d', "--depth", type=float, default=1.0)
    parser.add_argument('-r', "--resolution", type=float, default=32)
    args = parser.parse_args()

    inp_argument = (args.width, args.depth, args.resolution)

    env = NasEnv(n_train=45000, resolution = args.resolution)
    mem, acc, latency = env.eval_arg(inp_argument, print_model=True)
    print(f'acc: {acc:.2f}%, mem: {mem} MB')
