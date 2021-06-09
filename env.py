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
from latency_predictor import LatencyPredictor


def arg2model(inp_arg,
              modelName='preactresnet18',
              device='cuda',
              from_pretrained=False,
              _print=False):
    w, d, r = inp_arg
    path = "./" + str(w) + "_" + str(d) + "_" + str(r)

    if os.path.exists(path):
        if _print:
            print("Load existing model: " + path)
        model = torch.load(path)
    elif from_pretrained:
        if _print:
            print("Currently, no pretrained weight")
    else:
        if _print:
            print("Create new model: ", modelName)
        if 'preactresnet' in modelName:
            model = models.__dict__[modelName](100, False, 1, w, d).to(device)
        elif 'efficientnet' in modelName:
            model = models.__dict__["EfficientNet"].from_name("efficientnet-b0",
                                                              width_coefficient=w,
                                                              image_size=r,
                                                              depth_coefficient=d).to(device)
            final_filters = 1280
            new_filters = round_filters(final_filters, model._global_params)
            model._fc = torch.nn.Linear(new_filters, 100, bias=True).to(device)

    return model


def _print_model(model, resolution, device='cuda'):
    report, _ = logger.summary_string(model, (3, resolution, resolution),
                                      batch_size=1,
                                      device=device)
    print(report)


def latency_cal(dic, module, block, strided, image_size, latency_model, test_predictor):
    latency = 0
    predicted_latency = 0
    last_image_size = image_size
    if 'Block' in module.__class__.__name__:
        block = True
    if ('stride' in dir(module)) and not strided:
        image_size = max(1, image_size // module.stride[0])
        if block is True:
            strided = True
    has_children = False
    for child_name, child_module in module.named_children():
        image_size, strided, temp_latency, temp_predicted_latency = latency_cal(dic, child_module, block, strided,
                                                        image_size, latency_model, test_predictor)
        latency = latency + temp_latency
        predicted_latency = predicted_latency + temp_predicted_latency
        has_children = True
    if 'Block' in module.__class__.__name__:
        if repr((last_image_size, image_size, module)) in dic.keys():
            latency = latency + dic[repr((last_image_size, image_size, module))]['value']
            if test_predictor:
                predicted_latency = predicted_latency + latency_model.predict(last_image_size, image_size, module)
        else:
            latency = latency + latency_model.predict(last_image_size, image_size, module)
            #print("no such key: ", repr((last_image_size, image_size, module)))
        strided = False

    if has_children is False and block is False:
        if repr((last_image_size, image_size, module)) in dic.keys():
            latency = latency + dic[repr((last_image_size, image_size, module))]['value']
            if test_predictor:
                predicted_latency = predicted_latency + latency_model.predict(last_image_size, image_size, module)
        else:
            latency = latency + latency_model.predict(last_image_size, image_size, module)
            #print("no such key: ", repr((last_image_size, image_size, module)))
        strided = False

    return image_size, strided, latency, predicted_latency

class NasEnv():
    def __init__(self,
                 modelName='preactresnet18',
                 platform='desktop',
                 device='cuda',
                 batch_size=64,
                 lr=1e-2,
                 n_epoch=30
                 ):
        self.modelName = modelName
        self.platform = platform
        self.batch_size = batch_size
        self.lr = lr
        if device is 'default':
            self.device = "cpu"
        else:
            self.device = device

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.eval_fn = accuracy
        self.epoch = n_epoch
        self.latency_model = LatencyPredictor(platform=self.platform, device=self.device)

    def eval_arg(self, inp_arg, print_model=False, eval_acc=True):
        if torch.is_tensor(inp_arg):
            inp_arg = inp_arg.cpu().detach().numpy()

        resolution = int(inp_arg[-1])  # resolution
        self.trainloader, self.validloader, self.testloader = dataloader(
            batch_size=self.batch_size, input_resolution=resolution, _print=print_model)

        print("\nStart evaluating ", inp_arg)
        model = arg2model(inp_arg, modelName=self.modelName, device=self.device, _print=print_model)
        if print_model:
            _print_model(model, resolution, device=self.device)

        mem = self.eval_mem(model, resolution)
        latency = self.eval_latency(model)

        if eval_acc:
            acc = self.eval_acc(model)
            print(f'acc: {acc:.2f}%, mem: {mem}MB, latency: {latency:.2f}us')
            return mem, acc, latency
        else:
            print(f'mem: {mem}MB, latency: {latency:.2f}us')
            return mem, latency

    def eval_mem(self, model, resolution):
        report, _ = logger.summary_string(model, (3, resolution, resolution),
                                          batch_size=1,
                                          device=self.device)

        estimated_mem = float(report.split('\n')[-3].split(' ')[-1])  # (MB)
        return estimated_mem

    def eval_latency(self, model, resolution, test_predictor = False):
        path = ('latency_data/' + self.platform + '/' + self.modelName + '/' + self.device +
                '/image_' + str(resolution) + '.yaml')
        latency = 0
        dic = {}
        try:
            with open(path, 'r') as f:
                dic = yaml.load(f, Loader=yaml.FullLoader)
                for mod in model.modules():
                    _, _, latency, predicted_latency = latency_cal(dic, mod, False, False,
                            resolution, self.latency_model, test_predictor = test_predictor)
                    break
            #print("model latency is ", latency, " us")
        except Exception as ex:
            for mod in model.modules():
                _, _, latency, predicted_latency = latency_cal(dic, mod, False, False,
                        resolution, self.latency_model, test_predictor = test_predictor)
                break

            #print(path,": no such file")
        if test_predictor == True:
            err = (latency - predicted_latency) / latency * 100
            print("latency: {} us, predicted: {} us, err {}%".format(latency, predicted_latency, err))
        return latency

    def eval_acc(self, model):
        """
        Make model according to the input argument and return the accuracy and memory of the model.
        Args
        inp_arg: tuple(width_scale, depth_scale, resolution) e.g., (1.0, 1.0, 32)

        Return
        accuracy, memory of the model
        """
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.epoch // 2, gamma=0.3)

        best_test_acc = 0

        for i in range(self.epoch):
            s = time()
            train_top1, _, train_loss = self._train_epoch(model, optimizer, i)
            valid_top1, _, valid_loss = self._valid(model)

            if valid_top1 > best_test_acc:
                best_test_acc = valid_top1

            if self.epoch > 1:
                scheduler.step()

        print(
            f'[epoch {i+1} ({time()-s:.2f})] (train) loss {train_loss:.2f}, top1 {train_top1:.2f}%',
            end=' | ')
        print(f'(valid) loss = {valid_loss:.2f}, top1 = {valid_top1:.2f}%')
        return best_test_acc

    def _train_epoch(self, model, optimizer, i):
        top1, top5, loss = train(self.trainloader, model, optimizer, self.device, self.loss_fn,
                                 self.eval_fn, i)
        return top1, top5, loss

    def _valid(self, model):
        top1, top5, loss = validate(self.validloader, model, self.device, self.loss_fn,
                                    self.eval_fn)
        return top1, top5, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''Environment'''
    parser.add_argument('-m', "--model", type=str, default='preactresnet18')
    parser.add_argument('-p', "--platform", type=str, default='desktop')
    parser.add_argument('-v', "--device", type=str, default='cuda')
    parser.add_argument('-w', "--width", type=float, default=1.0)
    parser.add_argument('-d', "--depth", type=float, default=1.0)
    parser.add_argument('-r', "--resolution", type=int, default=32)
    parser.add_argument('-e', "--epoch", type=int, default=1)
    args = parser.parse_args()

    inp_argument = (args.width, args.depth, args.resolution)

    env = NasEnv(modelName=args.model,
                 platform=args.platform,
                 device=args.device,
                 n_epoch=args.epoch)
    mem, acc, latency = env.eval_arg(inp_argument, print_model=True)
    print(f'acc: {acc:.2f}%, mem: {mem}MB, latency: {latency:.2f}us')
