import models
import os.path as os_path
import torch
import parse_yaml

model_name = "preactresnet18"
model = models.__dict__[model_name](100, False, 1, 30 / 10, 1.0).to('cpu')


class LatencyPredictor():
    def __init__(self, platform = "desktop", device= "cpu", load_path = "./trained_model/"):
        if device == "cuda":
            device = "gpu"

        modelnames = ["block_0", "block_1", "block_2", "block_3"]
        filenames = [name+".pt" for name in modelnames]
        dirname = "{}_{}".format(platform,device)
        load_paths = [os_path.join(load_path,dirname,filename) for filename in filenames]
        self.models = {name:torch.load(load_path) for load_path, name in zip(load_paths, modelnames)}
        for name in self.models:
            self.models[name].eval()
    
    def predict(self, last_image_size, image_size, module, device = "cuda"):
        x, block_type = parse_yaml.parse(repr((last_image_size, image_size, module)))
        model = self.models[block_type]
        x = torch.FloatTensor(x).to(device)
        x_min = model.min.to(device)
        x_max = model.max.to(device)
        x = (x - x_min) / (x_max - x_min)
        y = model(x)
        return y.item()


def latency_model(model, image_size):
    model_type = model.__class__.__name__
    if model_type != "PreActResNet":
        print("Only PreActResNet Available")
        return
    latency = 0
    for a in model.children():
        module_latency = predict_latency_module(a, image_size)
        latency = latency + module_latency
    return latency

def predict_latency_module(module, image_size):
    module_type = type(module).__name__
    if module_type == "Sequential":
        return sequential_latency(module, image_size)

    elif module_type == "Conv2d":
        return conv2d_latency(module, image_size)
    
    elif module_type == "Linear":
        return linear_latency(module, image_size)
    
    else:
        print("Error, such module not available")
        return 0

def conv2d_latency(module, image_size):
    return 0

def linear_latency(module, image_size):
    return 0

def sequential_latency(module, image_size):
    return 0

#print(predict_latency(model, 32))

