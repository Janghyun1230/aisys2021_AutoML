import argparse
import torch
import os
import logger
from time import time
import models
from trainer import train, validate, accuracy
from data import dataloader
from efficientnet_utils import round_filters
import yaml
import copy

def arg2model(inp_arg, modelName='preactresnet18', device='cuda', from_pretrained=False):
    w, d, r = inp_arg
    path = "./" + str(w) + "_" + str(d) + "_" + str(r)
    if os.path.exists(path):
        print("Load existing model: " + path)
        model = torch.load(path)
    elif from_pretrained:
        print("Currently, no pretrained weight")
    else:
        print("Create new model: ", modelName)
        if 'preactresnet' in modelName:
            model = models.__dict__[modelName](100, False, 1, w,
                                                      d).to(device)
        elif 'efficientnet' in modelName:
            model = models.__dict__["EfficientNet"].from_name("efficientnet-b0", 
                    width_coefficient=w, image_size=r, depth_coefficient=d).to(device)
            final_filters = 1280
            new_filters = round_filters(final_filters, model._global_params)
            model._fc = torch.nn.Linear(new_filters, 100,
                                        bias=True).to(device)
    return model


def _print_model(model, resolution, device='cuda'):
    report, _ = logger.summary_string(model, (3, resolution, resolution),
                                      batch_size=1,
                                      device=device)
    print(report)

def latency_cal(dic, module, block, strided, image_size):
    latency = 0
    last_image_size = image_size
    if 'Block' in module.__class__.__name__:
        block = True
    if ('stride' in dir(module)) and not strided:
        image_size = max(1,image_size//module.stride[0])
        if block is True:
            strided = True
    has_children = False
    for child_name, child_module in module.named_children():
        image_size, strided, temp_latency = latency_cal(dic, child_module, block, strided, image_size)
        latency = latency + temp_latency
        has_children = True
    if 'Block' in module.__class__.__name__:
        if repr((last_image_size, image_size, module)) in dic.keys():
            latency = latency + dic[repr((last_image_size, image_size, module))]['value']
        else:
            print("no such key: ", repr((last_image_size, image_size, module)))
        strided = False

    if has_children is False and block is False:
        if repr((last_image_size, image_size, module)) in dic.keys():
            latency = latency + dic[repr((last_image_size, image_size, module))]['value']
        else:
            print("no such key: ", repr((last_image_size, image_size, module)))
        strided = False
            
    return  image_size, strided, latency

class NasEnv():
    def __init__(self,
                 modelName='preactresnet18',
                 platform='desktop',
                 device='default',
                 batch_size=64,
                 lr=1e-2,
                 resolution=32,
                 n_train=45000,
                 n_epoch=30):
        self.trainloader, self.validloader, self.testloader = dataloader(
            batch_size=batch_size, input_resolution=resolution)
        self.modelName = modelName
        self.platform = platform
        self.batch_size = batch_size
        self.resolution = resolution
        self.lr = lr
        if device is 'default':
            self.device = "cpu"
        else:
            self.device = device

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.eval_fn = accuracy
        self.epoch = n_epoch

    def eval_arg(self, inp_arg, print_model=False):
        print("\nStart evaluating ", inp_arg)
        model = arg2model(inp_arg, modelName=self.modelName, device=self.device)
        if print_model:
            _print_model(model, self.resolution, device=self.device)
        mem = self.eval_mem(model)
        latency = self.eval_latency(model)
        acc = self.eval_acc(model)

        return mem, acc, latency

    def eval_mem(self, model):
        report, _ = logger.summary_string(
            model, (3, self.resolution, self.resolution),
            batch_size=self.batch_size,
            device=self.device)

        estimated_mem = float(report.split('\n')[-3].split(' ')[-1])  # (MB)
        return estimated_mem


    def eval_latency(self, model):
        path = ('latency_data/'+self.platform+'/' 
               + self.modelName + '/' + self.device 
               + '/image_' + str(self.resolution) + '.yaml')
        latency = 0
        try:
            with open(path, 'r') as f:
                dic = yaml.load(f, Loader=yaml.FullLoader)
                for mod in model.modules():
                    _1, _2, latency = latency_cal(dic, mod, False, False, self.resolution)
                    break
            print("model latency is ", latency ," us")
        except Exception as ex:
            print(ex)
            #print(path,": no such file")

        return latency

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
    parser.add_argument('-m', "--model", type=str, default='preactresnet18')
    parser.add_argument('-p', "--platform", type=str, default='desktop')
    parser.add_argument('-v', "--device", type=str, default='default')
    parser.add_argument('-w', "--width", type=float, default=1.0)
    parser.add_argument('-d', "--depth", type=float, default=1.0)
    parser.add_argument('-r', "--resolution", type=int, default=32)
    args = parser.parse_args()

    inp_argument = (args.width, args.depth, args.resolution)

    env = NasEnv(n_train=45000, modelName = args.model, 
            platform = args.platform, device = args.device, resolution = args.resolution)
    mem, acc, latency = env.eval_arg(inp_argument, print_model=True)
    print(f'acc: {acc:.2f}%, mem: {mem} MB, latency: {latency:.2f} us')
