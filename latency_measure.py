import torchprof
import argparse
from models.efficientnet import EfficientNet
from efficientnet_utils import round_filters
from efficientnet_utils import BlockDecoder
from efficientnet_utils import GlobalParams
from data import dataloader
import torch
from torchprof.display import _flatten_tree, _build_measure_tuple, group_by
import yaml
import models
import os

def latencyAdd(latency_dic, key, added_count, added_value):
    if 'str' not in key.__class__.__name__:
        key = repr(key)
    if key not in latency_dic.keys():
        latency_dic[key] = {'count': 0, 'value': 0}
    last_count = latency_dic[key]['count']
    last_value = latency_dic[key]['value']
    latency_dic[key]['count'] = last_count + added_count
    latency_dic[key]['value'] = (last_count * last_value + added_value)/(last_count + added_count)

def latencyTraces(traces, trace_events, image_size):
    latency_dic = {}
    temp_latency_dic = {}
    last_block = 'first'
    cur_block = None
    cur_module = None
    value = 0
    temp_image_size = image_size

    for trace in traces:
        [path, leaf, module] = trace
        if cur_block is not None and not (len([e for e in cur_block if e in path]) is len(cur_block)):
            latencyAdd(latency_dic, (last_image_size, image_size, cur_module), 1, value)
            temp_latency_dic = {}
            value = 0
            cur_block = None
        if 'Block' in module.__class__.__name__:
            cur_block = path
            cur_module = module
        if 'stride' in dir(module) and cur_block is not last_block:
            last_image_size = image_size
            image_size = max(1,image_size/module.stride[0])
            temp_last_image_size = last_image_size
            temp_image_size = image_size
            if cur_block is not None :
                last_block = cur_block
        if leaf:
            temp_value = 0
            events = [te for t_events in trace_events[path] for te in t_events]
            for event_name, event_group in group_by(
                        events, lambda e: e.name):
                event_group = list(event_group)
                measures = _build_measure_tuple(event_group,len(event_group))
                temp_value += measures.cpu_total if measures else 0
                temp_value += measures.cuda_total if measures else 0
            value = value + temp_value
            if cur_block is None:
                latencyAdd(latency_dic, (last_image_size, image_size, module), 1, value)
                temp_latency_dic = {}
                last_image_size = image_size
                value = 0
            else:
                latencyAdd(temp_latency_dic, (last_image_size, image_size, module), 1, temp_value)
                temp_last_image_size = temp_image_size
    for key in temp_latency_dic.keys():
        latencyAdd(latency_dic, key, temp_latency_dic[key]['count'], temp_latency_dic[key]['value'])

    return latency_dic


def latency_efficientnet(width_coefficient=None,
                         depth_coefficient=1.0,
                         image_size=None,
                         dropout_rate=0.2,
                         drop_connect_rate=0.2,
                         num_classes=1000,
                         include_top=True):
    blocks_args = [
        'r2_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r2_k3_s22_e6_i40_o80_se0.25',
        'r2_k5_s11_e6_i80_o112_se0.25',
        'r2_k5_s22_e6_i112_o192_se0.25',
        'r2_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        image_size=image_size,
        dropout_rate=dropout_rate,
        num_classes=num_classes,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        drop_connect_rate=drop_connect_rate,
        depth_divisor=8,
        min_depth=None,
        include_top=include_top,
    )

    return blocks_args, global_params


def latencyTest(model, dataloader, device, image):
    model.eval()
    if device == "cuda":
        model.cuda()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torchprof.Profile(model, use_cuda=True) as prof:
                logits = model(inputs)
                preds = logits.softmax(dim=1)
            if prof.exited:
                return latencyTraces(prof.traces,
                                     prof.trace_profile_events, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--platform", type=str, default="desktop")
    parser.add_argument('-d', "--device", type=str, default="cpu")
    parser.add_argument('-m', "--model", type=str, default="efficientnet")
    args = parser.parse_args()
    dir_path = 'latency_data/' + args.platform + '/' + args.model + '/' + args.device + '/'
    print(args)
    repetition = 20
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for image in range(32, 225, 4):
        trainloader, validloader, testloader = dataloader(
                batch_size=64, input_resolution=image, n_valid=9999)
        file_path = dir_path + 'image_' + str(image) + '.yaml'
        for wc in range(10, 51, 1):
            print("image: ", image, ", wc: ", wc)
            # make model
            if 'efficientnet' in args.model:
                block_args, global_params = latency_efficientnet(
                    width_coefficient=wc / 10, image_size=image)
                model = EfficientNet(block_args, global_params).to(args.device)
                new_filters = round_filters(1280, model._global_params)
                model._fc = torch.nn.Linear(new_filters, 100,
                                            bias=True).to(args.device)
            elif 'preactresnet' in args.model:
                model = models.__dict__[args.model](100, False, 1, wc / 10,
                                                    1.0).to(args.device)
            # latency measure
            for i in range(repetition):
                if repetition - 1 is i:
                    print("repetition : ", i+1)
                else:
                    print("repetition : ", i+1, end='\r')
                cur_latency_dic = latencyTest(model, testloader, args.device, image)
                if os.path.exists(file_path):
                    f = open(file_path, 'r')
                    latency_dic = yaml.load(f, Loader=yaml.FullLoader)
                    f.close()
                else:
                    latency_dic = {}
                for key in cur_latency_dic.keys():
                    latencyAdd(latency_dic, key, cur_latency_dic[key]['count'], cur_latency_dic[key]['value'])
                f = open(file_path, 'w')
                yaml.dump(latency_dic, f)
                f.flush()
                f.close()
                del latency_dic
