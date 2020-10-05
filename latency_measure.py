import torchprof
import argparse
from efficientnet import EfficientNet
from efficientnet_utils import round_filters
from efficientnet_utils import BlockDecoder
from efficientnet_utils import GlobalParams
from data import dataloader
import torch
from torchprof.display import _flatten_tree, _build_measure_tuple, group_by
from collections import OrderedDict
import yaml


def latencyTraces(model, traces, trace_events):
    tree = OrderedDict()
    for trace in traces:
        [path, leaf, module] = trace
        current_tree = tree
        # unwrap all of the events, in case model is called multiple times
        events = [te for t_events in trace_events[path] for te in t_events]
        for depth, name in enumerate(path, 1):
            if name not in current_tree:
                current_tree[name] = OrderedDict()
            if depth == len(path) and (leaf):
                for event_name, event_group in group_by(
                        events, lambda e: e.name):
                    event_group = list(event_group)
                    current_tree[name][event_name] = {
                        None: _build_measure_tuple(event_group,
                                                   len(event_group))
                    }
            current_tree = current_tree[name]
    tree_lines = _flatten_tree(tree)

    #format_lines = []
    candidates = {}
    for mod in model.modules():
        for chs, chm in mod.named_children():
            if "block" in chs:
                for bi, bm in chm.named_children():
                    candidates[bi] = bm
            else:
                candidates[chs] = chm
        break
    candidate = ''
    value = 0
    latency_dic = {}
    for idx, tree_line in enumerate(tree_lines):
        depth, name, measures = tree_line
        if name in candidates.keys() and int(depth) < 3:
            if candidate != '':
                latency_dic[repr(candidates[candidate])] = value
                value = 0
            candidate = name
        #cpu_total = measures.cpu_total if measures else None
        #cuda_total = measures.cuda_total if measures else None
        #format_lines.append([depth, name, cpu_total, cuda_total])
        value += measures.cpu_total if measures else 0
        value += measures.cuda_total if measures else 0
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


def latencyTest(model, dataloader, device):
    model.eval()
    if device == "cuda":
        model.cuda()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torchprof.Profile(model, use_cuda=True,
                                   profile_memory=True) as prof:
                logits = model(inputs)
                preds = logits.softmax(dim=1)

            if prof.exited:
                return latencyTraces(model, prof.traces,
                                     prof.trace_profile_events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--device", type=str, default="desktop")
    parser.add_argument('-p', "--processor", type=str, default="CPU")
    args = parser.parse_args()
    print(args)
    for image in range(32, 224, 4):
        latency_dic = {}
        for wc in range(10, 51, 1):
            block_args, global_params = latency_efficientnet(
                width_coefficient=wc / 10, image_size=image)
            model = EfficientNet(block_args, global_params).to(args.processor)
            final_filters = 1280
            new_filters = round_filters(final_filters, model._global_params)
            model._fc = torch.nn.Linear(new_filters, 100,
                                        bias=True).to(args.processor)
            trainloader, validloader, testloader = dataloader(
                batch_size=64, input_resolution=image, n_valid=9999)
            cur_latency_dic_list = []
            for i in range(5):
                cur_latency_dic_list.append(
                    latencyTest(model, testloader, args.processor))
            for k in cur_latency_dic_list[0].keys():
                for i in range(5):
                    if k in latency_dic.keys():
                        latency_dic[k]['value'] = (
                            (latency_dic[k]['value'] * latency_dic[k]['count'])
                            + cur_latency_dic_list[i][k]) / (
                                latency_dic[k]['count'] + 1)
                        latency_dic[k]['count'] += 1
                    else:
                        latency_dic[k] = {}
                        latency_dic[k]['value'] = cur_latency_dic_list[i][k]
                        latency_dic[k]['count'] = 1
            print("image: ", image, ", wc: ", wc)
        f = open(args.device + '/image_' + str(image) + '.yaml', 'w')
        yaml.dump(latency_dic, f)
        f.close()
