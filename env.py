import argparse
import torch
import os
import logger
from utils import round_filters
from time import time
from model import MBConvBlock
from model import EfficientNet
from trainer import train, test, accuracy
from data import dataloader


def arg2model(inp_arg, device='cuda', from_pretrained=False):
    w, d, r = inp_arg
    path = "./" + str(w) + "_" + str(d) + "_" + str(r)
    if os.path.exists(path):
        print("Load existing model: " + path)
        model = torch.load(path)
    elif from_pretrained:
        print("From pretrained")
        model = EfficientNet.from_pretrained('efficientnet-b0',
                                             width_coefficient=w,
                                             depth_coefficient=d)
    else:
        print("Create new model")
        model = EfficientNet.from_name('efficientnet-b0',
                                       width_coefficient=w,
                                       depth_coefficient=d).to(device)

    final_filters = 1280  # for efficientnet-b0
    new_filters = round_filters(final_filters,
                                model._global_params)  # for custom model
    model._fc = torch.nn.Linear(new_filters, 100, bias=True).to(device)

    return model


def _print_model(model, resolution, device='cuda'):
    report, _ = logger.summary_string(model, (3, resolution, resolution),
                                      batch_size=1,
                                      device=device)
    print(report)


class NasEnv():
    def __init__(self,
                 batch_size=64,
                 lr=0.1,
                 resolution=32,
                 n_train=45000,
                 n_epoch=30):
        self.trainloader, self.validloader, self.testloader = dataloader(
            batch_size=batch_size,
            input_resolution=resolution,
            n_train=n_train)
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
        latency = self.eval_latency(model)

        return mem, acc, latency

    def eval_mem(self, model):
        report, _ = logger.summary_string(
            model, (3, self.resolution, self.resolution),
            batch_size=1,
            device=self.device)

        estimated_mem = float(report.split('\n')[-3].split(' ')[-1])  # (MB)
        return estimated_mem

    def eval_latency(self, model):
        '''
        TODO
        '''
        return None

    def eval_acc(self, model):
        """
        Make model according to the input argument and return the accuracy and memory of the model.
        ** Currently, the resolution does not affect the model. **

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
            train_loss, train_accuracy = self._train_epoch(model, optimizer)
            valid_loss, valid_accuracy = self._valid(model)
            if valid_accuracy > best_test_acc:
                best_test_acc = valid_accuracy
            scheduler.step()
            print(
                f'[epoch {i} ({time()-s:.2f})] (train) loss {train_loss:.3f}, acc {train_accuracy:.2%}',
                end=' | ')
            print(
                f'(valid) loss = {valid_loss:.3f}, acc = {valid_accuracy:.2%}')
        return best_test_acc

    def _train_epoch(self, model, optimizer):
        train_loss, train_accuracy = train(model, optimizer, self.trainloader,
                                           self.device, self.loss_fn,
                                           self.eval_fn)
        return train_loss, train_accuracy

    def _valid(self, model):
        test_loss, test_accuracy = test(model, self.validloader, self.device,
                                        self.loss_fn, self.eval_fn)
        return test_loss, test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''Environment'''
    parser.add_argument('-w', "--width", type=float, default=1.0)
    parser.add_argument('-d', "--depth", type=float, default=1.0)
    parser.add_argument('-r', "--resolution", type=float, default=32)
    args = parser.parse_args()

    inp_argument = (args.width, args.depth, args.resolution)

    env = NasEnv(n_train=45000)
    mem, acc, latency = env.eval_arg(inp_argument, print_model=True)
    print(f'acc: {acc:.2f}%, mem: {mem} MB')
